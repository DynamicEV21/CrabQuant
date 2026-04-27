"""
Daemon state persistence — tracks cross-wave daemon progress via atomic JSON saves.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class DaemonState:
    """Persistent daemon-level state for multi-wave refinement runs."""

    daemon_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_wave_completed: str = ""
    current_wave: int = 0
    total_mandates_run: int = 0
    total_strategies_promoted: int = 0
    total_api_calls: int = 0
    pending_mandates: list = field(default_factory=list)
    completed_mandates: list = field(default_factory=list)
    failed_mandates: list = field(default_factory=list)
    last_error: Optional[str] = None
    shutdown_requested: bool = False

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls) -> "DaemonState":
        """Create fresh state with defaults."""
        return cls()

    # ── serialisation ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Atomic save: write to path.tmp then os.rename()."""
        data = asdict(self)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> Optional["DaemonState"]:
        """Load from JSON. Return None if missing or corrupted."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError):
            return None

    # ── mutations ──────────────────────────────────────────────────────────

    def heartbeat(self, path: str) -> None:
        """Update last_heartbeat to now, save."""
        self.last_heartbeat = datetime.now(timezone.utc).isoformat()
        self.save(path)

    def record_wave_start(self, wave_num: int, mandate_name: str, path: str) -> None:
        """Record start of a new wave, save."""
        self.current_wave = wave_num
        if mandate_name not in self.pending_mandates:
            self.pending_mandates.append(mandate_name)
        self.save(path)

    def record_wave_completion(
        self, mandate_name: str, status: str, sharpe: float, path: str
    ) -> None:
        """Record completion of a mandate, update counters, save."""
        self.total_mandates_run += 1
        self.last_wave_completed = datetime.now(timezone.utc).isoformat()
        self.last_error = None

        # Remove from pending
        if mandate_name in self.pending_mandates:
            self.pending_mandates.remove(mandate_name)

        if status == "success":
            self.completed_mandates.append(mandate_name)
            if sharpe >= 1.5:
                self.total_strategies_promoted += 1
        else:
            self.failed_mandates.append(mandate_name)

        self.save(path)

    def get_resume_point(self) -> Optional[str]:
        """Return next pending mandate filename, or None if empty."""
        return self.pending_mandates[0] if self.pending_mandates else None

    def mark_shutdown(self, path: str) -> None:
        """Set shutdown_requested=True, save."""
        self.shutdown_requested = True
        self.save(path)
