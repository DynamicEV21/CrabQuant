"""Action Effectiveness Analytics — Phase 6.

Mine run_history.jsonl to compute per-failure-mode action success rates,
providing data-driven action recommendations for the LLM refinement loop.

Unlike action_analytics.py which computes global stats, this module cross-references
failure_mode with action type to tell the LLM which actions historically work
best for each specific failure mode (e.g., "novel" works best for low_sharpe).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from crabquant.refinement.action_analytics import load_run_history

logger = logging.getLogger(__name__)

# Mandates that are test/smoke runs and should be excluded from analytics
SKIP_MANDATES = frozenset({
    "smoke_test",
    "test_mandate",
    "module_fail",
    "unreachable_target",
    "high_target",
    "history_test",
    "modify_test",
    "multi_ticker",
    "single_turn",
    "e2e_stress_test",
    "e2e_test_momentum",
})


def analyze_action_effectiveness(history_path: str | Path) -> dict[str, Any]:
    """Analyze action effectiveness per failure mode from run history.

    Reads run_history.jsonl, filters out test/smoke entries, and computes
    per-failure-mode action success rates.

    Args:
        history_path: Path to the run_history.jsonl file.

    Returns:
        Dict with structure:
        {
            "total_entries": int,           # total entries after filtering
            "by_failure_mode": {
                "low_sharpe": {
                    "total": int,            # entries with this failure mode
                    "actions": {
                        "novel": {"total": 16, "successes": 2, "success_rate": 0.125},
                        "replace_indicator": {"total": 5, "successes": 0, "success_rate": 0.0},
                    },
                    "ranked_actions": [      # sorted by success_rate desc, then total desc
                        ("novel", 0.125, 16, 2),
                        ("replace_indicator", 0.0, 5, 0),
                    ],
                },
                ...
            },
        }
    """
    history = load_run_history(str(history_path))

    # Filter out test/smoke mandates
    filtered = [
        entry for entry in history
        if entry.get("mandate", "") not in SKIP_MANDATES
    ]

    result: dict[str, Any] = {
        "total_entries": len(filtered),
        "by_failure_mode": {},
    }

    if not filtered:
        return result

    # Group by failure_mode -> action
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for entry in filtered:
        failure_mode = entry.get("failure_mode", "")
        action = entry.get("action", "unknown")
        if not failure_mode:
            continue  # skip successful entries (empty failure_mode)
        buckets[failure_mode][action].append(entry)

    # Compute stats per failure mode
    for failure_mode, actions in sorted(buckets.items()):
        mode_total = sum(len(entries) for entries in actions.values())
        action_stats: dict[str, dict[str, Any]] = {}

        for action, entries in actions.items():
            total = len(entries)
            successes = sum(1 for e in entries if e.get("success", False))
            action_stats[action] = {
                "total": total,
                "successes": successes,
                "success_rate": successes / total if total > 0 else 0.0,
            }

        # Rank actions: by success_rate desc, then total desc (for tiebreaking)
        ranked = sorted(
            action_stats.items(),
            key=lambda x: (x[1]["success_rate"], x[1]["total"]),
            reverse=True,
        )

        result["by_failure_mode"][failure_mode] = {
            "total": mode_total,
            "actions": action_stats,
            "ranked_actions": [
                (action, stats["success_rate"], stats["total"], stats["successes"])
                for action, stats in ranked
            ],
        }

    return result


def format_action_effectiveness_for_prompt(
    effectiveness_data: dict[str, Any],
    failure_mode: str,
) -> str:
    """Format action effectiveness data for a specific failure mode as prompt text.

    Produces a concise section showing which actions have historically worked
    best for the given failure mode, helping the LLM make data-driven choices.

    Args:
        effectiveness_data: Output from analyze_action_effectiveness().
        failure_mode: The failure mode to generate recommendations for.

    Returns:
        Formatted string for prompt injection. Returns empty string if no data
        is available for the failure mode.
    """
    total_entries = effectiveness_data.get("total_entries", 0)
    by_mode = effectiveness_data.get("by_failure_mode", {})
    mode_data = by_mode.get(failure_mode)

    if not mode_data or not mode_data.get("actions"):
        return ""

    lines = [
        f"### Action Effectiveness (from {total_entries} real attempts)",
        f"For {failure_mode} failures:",
    ]

    ranked = mode_data.get("ranked_actions", [])
    for action, success_rate, total, successes in ranked:
        pct = success_rate * 100
        suffix = ""
        if len(ranked) > 0 and ranked[0][0] == action and success_rate > 0:
            suffix = " — best so far"
        lines.append(f"- {action}: {successes}/{total} ({pct:.1f}%){suffix}")

    # Generate recommendation
    if ranked:
        # Recommend top 2 actions with non-zero success rate, or top 2 overall
        recommended = []
        for action, sr, total, successes in ranked:
            if sr > 0:
                recommended.append(action)
            if len(recommended) >= 2:
                break

        # If no actions with positive success rate, recommend the most-tried ones
        if not recommended:
            # Sort by total attempts desc for untried/zero-success cases
            by_total = sorted(ranked, key=lambda x: x[2], reverse=True)
            recommended = [a for a, _, _, _ in by_total[:2]]

        if len(recommended) == 1:
            rec_line = f"Recommended: start with `{recommended[0]}`"
        else:
            rec_line = f"Recommended: start with `{recommended[0]}` or `{recommended[1]}`"
        lines.append(rec_line)

    return "\n".join(lines)
