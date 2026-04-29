"""Action Analytics — Phase 3.

Track which LLM action types succeed vs fail across refinement runs.
Aggregate stats, compute success rates, and generate context for the LLM
so it can learn from historical action outcomes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path for the run history JSONL file
RUN_HISTORY_FILE = str(Path("results/run_history.jsonl"))


# Backwards-compatible alias
ActionStats = dict


def load_run_history(path: str | None = None) -> list[dict]:
    """Load run history entries from a JSONL file.

    Each line is a JSON object. Malformed lines are silently skipped.
    Missing files return an empty list.

    Args:
        path: Path to the JSONL file. Defaults to RUN_HISTORY_FILE.

    Returns:
        List of history entry dicts.
    """
    path = path or RUN_HISTORY_FILE
    p = Path(path)
    if not p.exists():
        return []

    entries: list[dict] = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError as e:
        logger.warning("Failed to read run history %s: %s", path, e)

    return entries


def track_action_result(
    mandate: str,
    turn: int,
    action: str,
    sharpe: float,
    success: bool,
    failure_mode: str = "",
    path: str | None = None,
    error_info: dict | None = None,
) -> None:
    """Append an action result entry to the run history JSONL file.

    Args:
        mandate: Mandate name.
        turn: Turn number within the mandate.
        action: Action type (e.g., "modify_params", "full_rewrite").
        sharpe: Sharpe ratio achieved.
        success: Whether the turn met the Sharpe target.
        failure_mode: Failure classification (empty if success).
        path: Path to the JSONL file. Defaults to RUN_HISTORY_FILE.
        error_info: Optional dict with error_type, error_message, error_traceback
                    for backtest_crash and module_load_failed failures.
    """
    path = path or RUN_HISTORY_FILE
    entry = {
        "mandate": mandate,
        "turn": turn,
        "action": action,
        "sharpe": sharpe,
        "success": success,
        "failure_mode": failure_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error_info is not None:
        entry["error_type"] = error_info.get("error_type", "unknown")
        entry["error_message"] = str(error_info.get("error_message", ""))[:500]
        entry["error_traceback"] = str(error_info.get("error_traceback", ""))[:1000]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def aggregate_action_stats(history: list[dict]) -> dict[str, dict[str, Any]]:
    """Aggregate statistics per action type from run history.

    Args:
        history: List of history entry dicts (from ``load_run_history``).

    Returns:
        Dict mapping action name to stats dict with keys:
        total, successes, failures, success_rate, avg_sharpe.
    """
    if not history:
        return {}

    buckets: dict[str, list[dict]] = {}
    for entry in history:
        action = entry.get("action", "unknown")
        buckets.setdefault(action, []).append(entry)

    stats: dict[str, dict[str, Any]] = {}
    for action, entries in buckets.items():
        total = len(entries)
        successes = sum(1 for e in entries if e.get("success", False))
        sharpes = [e.get("sharpe", 0.0) for e in entries]
        avg_sharpe = sum(sharpes) / total if total > 0 else 0.0

        stats[action] = {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_sharpe": round(avg_sharpe, 4),
        }

    return stats


def compute_action_success_rates(
    history: list[dict],
) -> list[dict[str, Any]]:
    """Compute action success rates, sorted by success_rate descending.

    Args:
        history: List of history entry dicts.

    Returns:
        List of dicts with keys: action, success_rate, total, successes, avg_sharpe.
        Sorted by success_rate descending.
    """
    stats = aggregate_action_stats(history)
    if not stats:
        return []

    rates = []
    for action, s in stats.items():
        rates.append({
            "action": action,
            "success_rate": s["success_rate"],
            "total": s["total"],
            "successes": s["successes"],
            "avg_sharpe": s["avg_sharpe"],
        })

    rates.sort(key=lambda x: x["success_rate"], reverse=True)
    return rates


def generate_llm_context(history: list[dict]) -> str:
    """Generate a text summary of action analytics for LLM context injection.

    Args:
        history: List of history entry dicts.

    Returns:
        Formatted string with action success rates for LLM consumption.
    """
    if not history:
        return "No historical action data available. This is the first run."

    rates = compute_action_success_rates(history)
    if not rates:
        return "No actionable data from previous runs."

    lines = ["Historical action success rates (most successful first):"]
    for entry in rates:
        pct = entry["success_rate"] * 100
        lines.append(
            f"  - {entry['action']}: {entry['successes']}/{entry['total']} "
            f"({pct:.0f}%), avg Sharpe: {entry['avg_sharpe']:.2f}"
        )

    # Add recommendation
    best = rates[0]
    worst = rates[-1]
    lines.append(
        f"\nRecommendation: '{best['action']}' has the highest success rate. "
        f"'{worst['action']}' has the lowest — consider alternative approaches."
    )

    return "\n".join(lines)
