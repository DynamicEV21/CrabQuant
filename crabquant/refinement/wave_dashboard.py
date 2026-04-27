"""Wave Dashboard — Phase 3.

Real-time dashboard for CrabQuant refinement waves. Provides visibility into:
- Running mandates and their progress
- Convergence rates across waves
- Best strategies by Sharpe ratio
- Overall wave progress

Outputs JSON for easy consumption by monitoring tools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DashboardSnapshot:
    """Point-in-time snapshot of wave dashboard state."""

    running_count: int = 0
    total_mandates: int = 0
    convergence_rate: float = 0.0
    wave_progress: float = 0.0
    best_strategies: list[dict[str, Any]] = field(default_factory=list)
    running_mandates: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _load_all_states(runs_dir: str | Path) -> list[dict]:
    """Load all state.json files from a runs directory.

    Args:
        runs_dir: Directory containing run subdirectories with state.json.

    Returns:
        List of parsed state dicts.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return []

    states: list[dict] = []
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        state_file = run_dir / "state.json"
        if not state_file.exists():
            continue
        try:
            state = json.loads(state_file.read_text())
            states.append(state)
        except (json.JSONDecodeError, OSError):
            continue

    return states


def collect_running_mandates(runs_dir: str | Path) -> list[dict]:
    """Find all mandates currently in 'running' status.

    Args:
        runs_dir: Directory containing run subdirectories.

    Returns:
        List of state dicts with status == 'running'.
    """
    states = _load_all_states(runs_dir)
    return [s for s in states if s.get("status") == "running"]


def compute_convergence_rate(states: list[dict]) -> float:
    """Compute the fraction of completed mandates that succeeded.

    Running mandates are excluded from the denominator.

    Args:
        states: List of state dicts.

    Returns:
        Convergence rate as a float 0.0–1.0.
    """
    completed = [s for s in states if s.get("status") not in ("running", "pending")]
    if not completed:
        return 0.0

    successes = sum(1 for s in completed if s.get("status") == "success")
    return successes / len(completed)


def compute_wave_progress(states: list[dict]) -> float:
    """Compute overall wave progress as the average turn fraction.

    Completed mandates count as 100%. Running mandates count as
    current_turn / max_turns. Pending mandates count as 0%.

    Args:
        states: List of state dicts.

    Returns:
        Progress as a float 0.0–1.0.
    """
    if not states:
        return 0.0

    total_progress = 0.0
    for state in states:
        status = state.get("status", "")
        if status in ("success", "failed", "max_turns_exhausted", "abandoned"):
            total_progress += 1.0
        elif status == "running":
            current = state.get("current_turn", 0)
            max_turns = state.get("max_turns", 7)
            if max_turns > 0:
                total_progress += current / max_turns
        # pending: 0.0

    return total_progress / len(states)


def find_best_strategies(
    states: list[dict],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Find the top strategies by best Sharpe ratio.

    Args:
        states: List of state dicts.
        top_n: Number of strategies to return.

    Returns:
        List of dicts with mandate_name and best_sharpe, sorted descending.
    """
    sorted_states = sorted(
        states,
        key=lambda s: s.get("best_sharpe", -999.0),
        reverse=True,
    )

    results = []
    for state in sorted_states[:top_n]:
        results.append({
            "mandate_name": state.get("mandate_name", ""),
            "best_sharpe": state.get("best_sharpe", 0.0),
            "best_turn": state.get("best_turn", 0),
            "status": state.get("status", ""),
        })

    return results


def generate_dashboard(runs_dir: str | Path) -> DashboardSnapshot:
    """Generate a full dashboard snapshot from the runs directory.

    Args:
        runs_dir: Directory containing run subdirectories.

    Returns:
        DashboardSnapshot with current wave state.
    """
    states = _load_all_states(runs_dir)
    running = collect_running_mandates(runs_dir)

    return DashboardSnapshot(
        running_count=len(running),
        total_mandates=len(states),
        convergence_rate=compute_convergence_rate(states),
        wave_progress=compute_wave_progress(states),
        best_strategies=find_best_strategies(states),
        running_mandates=[
            {
                "mandate_name": s.get("mandate_name", ""),
                "current_turn": s.get("current_turn", 0),
                "max_turns": s.get("max_turns", 7),
                "best_sharpe": s.get("best_sharpe", 0.0),
            }
            for s in running
        ],
    )


def snapshot_to_json(snapshot: DashboardSnapshot, **kwargs) -> str:
    """Serialize a DashboardSnapshot to JSON.

    Args:
        snapshot: DashboardSnapshot to serialize.
        **kwargs: Additional kwargs passed to json.dumps.

    Returns:
        JSON string.
    """
    return json.dumps(asdict(snapshot), **kwargs)
