"""
Stagnation detection for CrabQuant refinement pipeline.

Implements scoring formula based on consecutive failed turns, Sharpe plateau,
and lack of improvement.  Response protocol -- pivot, broaden, nuclear rewrite,
or abandon.
"""

from __future__ import annotations

import numpy as np


def compute_stagnation(history: list[dict]) -> tuple[float, str]:
    """Return (score: 0.0-1.0, trend: "improving"|"flat"|"declining").

    Score 0.0 = great progress, 1.0 = completely stuck.

    Factors (weighted):
      0.4  Sharpe trend  (linear-regression slope)
      0.3  Variance      (std of last-3 Sharpes)
      0.3  Repetition    (unique actions in last-3 turns)
    """
    if len(history) < 2:
        return (0.0, "improving")

    sharpes = [h["sharpe"] for h in history]

    # Factor 1: Sharpe trend (linear regression slope)
    slope = float(np.polyfit(range(len(sharpes)), sharpes, 1)[0])
    if slope < -0.1:
        trend_score, trend = 1.0, "declining"
    elif slope < 0.0:
        trend_score, trend = 0.8, "flat"
    else:
        trend_score, trend = 0.1, "improving"

    # Factor 2: Variance (oscillation = bad)
    variance = float(np.var(sharpes)) if len(sharpes) >= 3 else 0.0
    if variance > 0.08:
        variance_score = 1.0
    elif variance > 0.04:
        variance_score = 0.8
    elif variance > 0.02:
        variance_score = 0.5
    else:
        variance_score = 0.0

    # Factor 3: Action repetition
    actions = [h.get("action", "") for h in history[-3:]]
    if len(actions) >= 3 and all(a == "modify_params" for a in actions):
        repetition_score = 1.0
    elif len(actions) >= 2 and actions.count("modify_params") >= 2:
        repetition_score = 0.7
    else:
        repetition_score = 0.0

    # Weighted combination
    score = 0.4 * trend_score + 0.2 * variance_score + 0.4 * repetition_score
    return (round(score, 3), trend)


_FAILURE_ACTION_MISMATCHES: dict[tuple[str, str], str] = {
    ("too_few_trades", "change_exit_logic"): "Tightening exits won't create more trades",
    ("excessive_drawdown", "modify_params"): "Tweaks rarely fix drawdown",
}


def check_hypothesis_failure_alignment(
    failure_mode: str,
    addresses_failure: str,
    action: str,
) -> list[str]:
    """Warn when the LLM's proposed action is a poor fit for the diagnosed failure."""
    warnings: list[str] = []
    if failure_mode != addresses_failure:
        warnings.append(
            f"Diagnosed '{failure_mode}' but change addresses '{addresses_failure}'."
        )
    key = (failure_mode, action)
    if key in _FAILURE_ACTION_MISMATCHES:
        warnings.append(_FAILURE_ACTION_MISMATCHES[key])
    return warnings


def get_stagnation_response(iteration: int, score: float) -> dict:
    """Return prompt adjustments based on stagnation level.

    Order matters: abandon > pivot > nuclear > broaden > normal.
    """
    if iteration <= 3:
        return {"constraint": "normal", "prompt_suffix": ""}

    if score > 0.8:
        return {
            "constraint": "abandon",
            "prompt_suffix": "ABANDON: This strategy cannot converge. Archive it.",
        }

    if score > 0.7:
        return {
            "constraint": "pivot",
            "prompt_suffix": (
                "PIVOT: Your incremental changes are not working. Try adding a "
                "regime filter, switching entry signal archetype, or changing the "
                "exit logic entirely."
            ),
        }

    if score > 0.6 and iteration >= 6:
        return {
            "constraint": "nuclear",
            "prompt_suffix": (
                "NUCLEAR REWRITE: You must use a completely different indicator family "
                "and signal archetype. Do NOT modify existing parameters."
            ),
        }

    if score > 0.5:
        return {
            "constraint": "broaden",
            "prompt_suffix": (
                "BROADEN: You may change entry AND exit logic simultaneously. "
                "Consider structural changes, not just parameter tweaks."
            ),
        }

    return {"constraint": "normal", "prompt_suffix": ""}
