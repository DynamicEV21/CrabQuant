"""
Wave scaling — increase parallel limit to 5, add wave status tracking.

Extends the existing wave_manager.py with:
- Configurable parallel limits (default 5, max 10)
- Wave status tracking (pending/running/completed/failed)
- JSON persistence for wave status across runs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Configuration ────────────────────────────────────────────────────────────

SCALING_CONFIG: dict = {
    "default_parallel_limit": 5,
    "max_parallel_limit": 10,
    "status_file": "refinement/runs/wave_status.json",
}


# ── Parallel limit ───────────────────────────────────────────────────────────

def get_parallel_limit(override: int | None = None) -> int:
    """Get the parallel limit for wave execution.

    Args:
        override: Explicit value. Clamped to [1, max_parallel_limit].

    Returns:
        Integer parallel limit.
    """
    if override is not None:
        return max(1, min(int(override), SCALING_CONFIG["max_parallel_limit"]))
    return SCALING_CONFIG["default_parallel_limit"]


# ── Wave status dataclass ────────────────────────────────────────────────────

@dataclass
class WaveStatus:
    """Status of a single wave."""

    wave_number: int
    status: str = "pending"       # pending | running | completed | failed
    mandate_count: int = 0
    completed_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    started_at: str = ""
    completed_at: str = ""
    error: str = ""

    @property
    def convergence_rate(self) -> float:
        return self.successful_count / max(self.mandate_count, 1)

    def to_dict(self) -> dict:
        return {
            "wave_number": self.wave_number,
            "status": self.status,
            "mandate_count": self.mandate_count,
            "completed_count": self.completed_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "convergence_rate": self.convergence_rate,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WaveStatus":
        return cls(
            wave_number=d["wave_number"],
            status=d.get("status", "pending"),
            mandate_count=d.get("mandate_count", 0),
            completed_count=d.get("completed_count", 0),
            successful_count=d.get("successful_count", 0),
            failed_count=d.get("failed_count", 0),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
            error=d.get("error", ""),
        )


# ── Wave status tracker ─────────────────────────────────────────────────────

class WaveStatusTracker:
    """Tracks status of all waves in a session."""

    def __init__(self) -> None:
        self.waves: dict[int, WaveStatus] = {}
        self.current_wave: int = 0

    def start_wave(self, wave_number: int, mandate_count: int) -> None:
        """Mark a wave as started."""
        self.current_wave = wave_number
        self.waves[wave_number] = WaveStatus(
            wave_number=wave_number,
            status="running",
            mandate_count=mandate_count,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def complete_wave(
        self, wave_number: int, successful: int, failed: int
    ) -> None:
        """Mark a wave as completed."""
        if wave_number not in self.waves:
            return
        ws = self.waves[wave_number]
        ws.status = "completed"
        ws.successful_count = successful
        ws.failed_count = failed
        ws.completed_count = successful + failed
        ws.completed_at = datetime.now(timezone.utc).isoformat()

    def fail_wave(self, wave_number: int, error: str) -> None:
        """Mark a wave as failed."""
        if wave_number not in self.waves:
            return
        ws = self.waves[wave_number]
        ws.status = "failed"
        ws.error = error
        ws.completed_at = datetime.now(timezone.utc).isoformat()

    def get_status_summary(self) -> dict:
        """Return a summary dict of all wave statuses."""
        total_successful = sum(w.successful_count for w in self.waves.values())
        total_failed = sum(w.failed_count for w in self.waves.values())
        total_mandates = sum(w.mandate_count for w in self.waves.values())

        return {
            "total_waves": len(self.waves),
            "current_wave": self.current_wave,
            "total_mandates": total_mandates,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_convergence_rate": (
                total_successful / max(total_mandates, 1)
            ),
            "waves": {
                str(k): v.to_dict() for k, v in self.waves.items()
            },
        }

    def save(self, path: str) -> None:
        """Persist status to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.get_status_summary(), indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> "WaveStatusTracker":
        """Load status from a JSON file. Returns empty tracker if file doesn't exist."""
        tracker = cls()
        p = Path(path)
        if not p.exists():
            return tracker

        try:
            data = json.loads(p.read_text())
            for wave_key, wave_data in data.get("waves", {}).items():
                wave_num = int(wave_key)
                tracker.waves[wave_num] = WaveStatus.from_dict(wave_data)
                if wave_num > tracker.current_wave:
                    tracker.current_wave = wave_num
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # Return empty tracker on corrupt file

        return tracker


# ── Convenience ──────────────────────────────────────────────────────────────

def get_wave_status_summary(tracker: WaveStatusTracker) -> dict:
    """Standalone function to get a summary from a tracker."""
    return tracker.get_status_summary()
