"""
Discovery and cron health tracking for the daily brief.

Reads cron results, winners, promotions, retirements, and cron status.
"""

import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = None  # lazily resolved


def _get_results_dir() -> Path:
    """Resolve the CrabQuant results directory."""
    global RESULTS_DIR
    if RESULTS_DIR is None:
        RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
    return RESULTS_DIR


def get_recent_winners(hours: int = 24) -> dict:
    """
    Count winners found in the last N hours.

    Returns:
        Dict with keys: count, top_winners (list), total_tested
    """
    results_dir = _get_results_dir()
    winners_file = results_dir / "winners" / "winners.json"
    cron_log = results_dir / "logs" / "cron_results.jsonl"
    cron_state_file = results_dir / "cron_state.json"

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Count recent winners from winners.json (has 'discovered' timestamps)
    recent_count = 0
    top_winners = []

    if winners_file.exists():
        with open(winners_file) as f:
            winners = json.load(f)

        recent = []
        for w in winners:
            discovered = w.get("discovered", "")
            if not discovered:
                continue
            try:
                dt = datetime.fromisoformat(discovered)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    recent.append(w)
            except (ValueError, TypeError):
                continue

        recent_count = len(recent)

        # Get top 3 by score
        recent.sort(key=lambda w: w.get("score", 0), reverse=True)
        for w in recent[:3]:
            top_winners.append({
                "ticker": w.get("ticker", "?"),
                "strategy": w.get("strategy", "?"),
                "sharpe": round(w.get("sharpe", 0), 2),
                "return": round(w.get("return", 0) * 100, 1),
            })

    # Total combos tested from cron_state
    total_tested = 0
    if cron_state_file.exists():
        with open(cron_state_file) as f:
            state = json.load(f)
        total_tested = len(state.get("completed_combos", []))

    return {
        "count": recent_count,
        "top_winners": top_winners,
        "total_tested": total_tested,
    }


def get_recent_promotions(hours: int = 24) -> int:
    """
    Count strategies promoted to production in the last N hours.

    Returns:
        Number of promotions in the time window.
    """
    results_dir = _get_results_dir()
    registry_file = results_dir.parent / "strategies" / "production" / "registry.json"
    confirmed_file = results_dir / "confirmed" / "confirmed.json"

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Check confirmed.json for recent confirmations (ROBUST verdicts)
    count = 0
    if confirmed_file.exists():
        with open(confirmed_file) as f:
            confirmed = json.load(f)

        for entry in confirmed:
            confirmed_at = entry.get("confirmed_at", "")
            if not confirmed_at:
                continue
            try:
                dt = datetime.fromisoformat(confirmed_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff and entry.get("verdict") == "ROBUST":
                    count += 1
            except (ValueError, TypeError):
                continue

    # Also check registry.json
    if registry_file.exists():
        with open(registry_file) as f:
            registry = json.load(f)

        for entry in registry:
            promoted_at = entry.get("promoted_at", "")
            if not promoted_at:
                continue
            try:
                dt = datetime.fromisoformat(promoted_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    count += 1
            except (ValueError, TypeError):
                continue

    return count


def get_retirements(hours: int = 24) -> int:
    """
    Count strategies that failed re-validation in the last N hours.

    Returns:
        Number of retirements in the time window.
    """
    results_dir = _get_results_dir()
    confirmed_file = results_dir / "confirmed" / "confirmed.json"

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    count = 0
    if confirmed_file.exists():
        with open(confirmed_file) as f:
            confirmed = json.load(f)

        for entry in confirmed:
            confirmed_at = entry.get("confirmed_at", "")
            if not confirmed_at:
                continue
            try:
                dt = datetime.fromisoformat(confirmed_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff and entry.get("verdict") == "FAILED":
                    count += 1
            except (ValueError, TypeError):
                continue

    return count


def get_cron_status() -> dict:
    """
    Check health of CrabQuant cron jobs.

    Returns:
        Dict with keys: active, total, details (list)
    """
    try:
        result = subprocess.run(
            ["openclaw", "cron", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Command unavailable — assume 4 crons exist but report unknown status
        return {"active": 0, "total": 4, "details": []}

    # Parse output — look for lines mentioning crabquant
    lines = output.strip().split("\n")
    crabquant_lines = [l for l in lines if "crabquant" in l.lower() or "sweep" in l.lower() or "validate" in l.lower()]

    # Also count total crons and active ones from the full output
    total = 0
    active = 0

    for line in lines:
        # Look for indicators of cron entries
        if "ID:" in line or "cron" in line.lower():
            total += 1
        if "running" in line.lower() or "active" in line.lower() or "idle" in line.lower():
            active += 1

    # If we couldn't parse well, at least report what we found
    if total == 0:
        if len(crabquant_lines) > 0:
            total = len(crabquant_lines)
            active = total
        else:
            total = 4  # assume 4 crons

    return {
        "active": active,
        "total": total,
        "details": crabquant_lines[:4],
    }
