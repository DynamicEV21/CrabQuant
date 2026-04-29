"""Resource Limiter — monitors CPU/RAM/disk and recommends parallel worker counts.

Provides dynamic scaling of parallel workers based on available system
resources, with a /proc/meminfo fallback for environments where psutil
is unavailable or misbehaving.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
_GB = 1024**3
_PROC_MEMINFO = Path("/proc/meminfo")


def _ram_from_proc_meminfo() -> float | None:
    """Read MemAvailable from /proc/meminfo (Linux fallback).

    Returns available RAM in **bytes**, or *None* if the file is
    unreadable or the key is missing.
    """
    try:
        text = _PROC_MEMINFO.read_text()
    except (OSError, PermissionError):
        return None

    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) * 1024  # kB → bytes
                except ValueError:
                    return None
    return None


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Immutable snapshot of current system resources."""

    cpu_percent: float
    ram_available_gb: float
    disk_free_gb: float


class ResourceLimiter:
    """Monitors CPU, RAM and disk; recommends a safe parallel worker count.

    Behaviour rules
    ---------------
    * **CPU > cpu_threshold** → reduce parallel by 1 (minimum 1).
    * **RAM < min_ram_gb** → ``should_pause()`` returns *True*,
      ``get_recommended_parallel()`` returns 0.
    * **RAM ∈ [min_ram_gb, 4 GB)** → cap parallel at
      ``max(1, max_parallel - 1)``.
    * **RAM ≥ 4 GB** → full parallel allowed (subject to CPU rule).
    * **Disk < min_disk_gb** → ``should_pause()`` returns *True*.
    """

    def __init__(
        self,
        min_ram_gb: float = 2.0,
        min_disk_gb: float = 1.0,
        max_parallel: int = 3,
        cpu_threshold: float = 0.85,
    ) -> None:
        if min_ram_gb <= 0:
            raise ValueError("min_ram_gb must be positive")
        if min_disk_gb <= 0:
            raise ValueError("min_disk_gb must be positive")
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if not (0.0 < cpu_threshold <= 1.0):
            raise ValueError("cpu_threshold must be in (0, 1]")

        self.min_ram_gb = min_ram_gb
        self.min_disk_gb = min_disk_gb
        self.max_parallel = max_parallel
        self.cpu_threshold = cpu_threshold

        self._last_snapshot: ResourceSnapshot | None = None

    # ── Resource sampling ──────────────────────────────────────────────

    def _get_ram_available_bytes(self) -> float:
        """Return available RAM in bytes, using psutil with /proc/meminfo fallback."""
        try:
            return float(psutil.virtual_memory().available)
        except Exception:
            logger.debug("psutil.virtual_memory() failed, trying /proc/meminfo")

        fallback = _ram_from_proc_meminfo()
        if fallback is not None:
            return fallback

        raise RuntimeError(
            "Cannot determine available RAM: psutil failed and "
            "/proc/meminfo is unreadable or missing"
        )

    def _get_disk_free_bytes(self) -> float:
        """Return free disk space in bytes."""
        try:
            return float(psutil.disk_usage("/").free)
        except Exception:
            try:
                stat = os.statvfs("/")
                return float(stat.f_bavail * stat.f_frsize)
            except OSError:
                raise RuntimeError("Cannot determine free disk space")

    def _get_cpu_percent(self) -> float:
        """Return CPU utilisation as a fraction in [0, 1]."""
        try:
            return psutil.cpu_percent(interval=0.5) / 100.0
        except Exception:
            logger.warning("psutil.cpu_percent() failed, assuming 0")
            return 0.0

    # ── Public API ─────────────────────────────────────────────────────

    def check_resources(self) -> dict[str, Any]:
        """Sample current system resources and cache the snapshot.

        Returns a dict with keys ``cpu_percent``, ``ram_free_gb``,
        ``disk_free_gb``, and ``status``.
        """
        cpu = self._get_cpu_percent()
        ram_bytes = self._get_ram_available_bytes()
        disk_bytes = self._get_disk_free_bytes()

        ram_gb = ram_bytes / _GB
        disk_gb = disk_bytes / _GB

        self._last_snapshot = ResourceSnapshot(
            cpu_percent=cpu,
            ram_available_gb=ram_gb,
            disk_free_gb=disk_gb,
        )

        # Determine status string
        if ram_gb < self.min_ram_gb:
            status = "ram_critical"
        elif disk_gb < self.min_disk_gb:
            status = "disk_critical"
        elif cpu > self.cpu_threshold:
            status = "cpu_constrained"
        elif ram_gb < 4.0:
            status = "ram_constrained"
        else:
            status = "ok"

        return {
            "cpu_percent": round(cpu, 4),
            "ram_free_gb": round(ram_gb, 2),
            "disk_free_gb": round(disk_gb, 2),
            "status": status,
        }

    def should_pause(self) -> bool:
        """Return *True* if execution should pause (RAM or disk too low).

        Triggers a ``check_resources()`` call if no recent snapshot exists.
        """
        if self._last_snapshot is None:
            self.check_resources()

        snap = self._last_snapshot
        assert snap is not None  # guaranteed by check_resources above

        if snap.ram_available_gb < self.min_ram_gb:
            return True
        if snap.disk_free_gb < self.min_disk_gb:
            return True
        return False

    def get_recommended_parallel(self) -> int:
        """Return the recommended parallel worker count.

        Rules (applied in priority order):
        1. RAM < min_ram_gb → 0 (should_pause)
        2. Disk < min_disk_gb → 0 (should_pause)
        3. CPU > cpu_threshold → max(1, max_parallel - 1)
        4. RAM < 4 GB → max(1, max_parallel - 1)
        5. Otherwise → max_parallel
        """
        if self._last_snapshot is None:
            self.check_resources()

        snap = self._last_snapshot
        assert snap is not None

        # Critical — zero workers
        if snap.ram_available_gb < self.min_ram_gb:
            return 0
        if snap.disk_free_gb < self.min_disk_gb:
            return 0

        parallel = self.max_parallel

        # CPU-constrained
        if snap.cpu_percent > self.cpu_threshold:
            parallel = max(1, parallel - 1)

        # RAM-constrained (but above min)
        if snap.ram_available_gb < 4.0:
            parallel = max(1, parallel - 1)

        return parallel

    def get_status_summary(self) -> dict[str, Any]:
        """Full status dict for reporting / logging."""
        if self._last_snapshot is None:
            self.check_resources()

        snap = self._last_snapshot
        assert snap is not None

        return {
            "cpu_percent": round(snap.cpu_percent, 4),
            "ram_free_gb": round(snap.ram_available_gb, 2),
            "disk_free_gb": round(snap.disk_free_gb, 2),
            "recommended_parallel": self.get_recommended_parallel(),
            "should_pause": self.should_pause(),
            "max_parallel": self.max_parallel,
            "min_ram_gb": self.min_ram_gb,
            "min_disk_gb": self.min_disk_gb,
            "cpu_threshold": self.cpu_threshold,
        }
