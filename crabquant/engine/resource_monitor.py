"""
Resource Monitor — CPU/RAM-aware throttling for parallel operations.

Monitors system resources and provides adaptive worker count recommendations.
Used by parallel.py to dynamically scale worker count based on available RAM
and CPU load. Prevents OOM kills and CPU thrashing during heavy backtesting.

Phase 6 prep item — resource-aware parallelism.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Resource thresholds ──────────────────────────────────────────────────

# Per-worker memory estimate (MB) — each backtest worker loads a DataFrame
# (~50MB for 2y daily data) plus indicator computations (~100MB).
DEFAULT_MEMORY_PER_WORKER_MB = 150.0

# Safety margins
DEFAULT_RAM_RESERVE_MB = 500.0  # Reserve for OS + other processes
DEFAULT_MAX_RAM_USAGE_PCT = 0.80  # Never use more than 80% of total RAM
DEFAULT_MAX_CPU_PCT = 0.90  # Leave 10% CPU for other processes

# Check interval for adaptive throttling (seconds)
DEFAULT_CHECK_INTERVAL_S = 5.0


@dataclass
class ResourceSnapshot:
    """Point-in-time system resource snapshot."""
    cpu_percent: float
    ram_total_mb: float
    ram_used_mb: float
    ram_available_mb: float
    ram_usage_pct: float
    disk_free_gb: float
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    cpu_count: int = 1
    timestamp: float = 0.0

    @property
    def ram_headroom_mb(self) -> float:
        """RAM available minus safety reserve."""
        return max(0, self.ram_available_mb - DEFAULT_RAM_RESERVE_MB)

    @property
    def max_workers_by_ram(self) -> int:
        """Max workers that fit in available RAM."""
        if DEFAULT_MEMORY_PER_WORKER_MB <= 0:
            return self.cpu_count
        return max(1, int(self.ram_headroom_mb / DEFAULT_MEMORY_PER_WORKER_MB))

    @property
    def max_workers_by_cpu(self) -> int:
        """Max workers based on CPU headroom."""
        headroom = max(0.1, 1.0 - (self.cpu_percent / 100.0))
        return max(1, int(self.cpu_count * headroom))

    @property
    def is_ram_constrained(self) -> bool:
        """Whether RAM is the bottleneck."""
        return (
            self.ram_usage_pct > DEFAULT_MAX_RAM_USAGE_PCT
            or self.ram_available_mb < DEFAULT_RAM_RESERVE_MB * 2
        )

    @property
    def is_cpu_constrained(self) -> bool:
        """Whether CPU is the bottleneck."""
        return self.cpu_percent > DEFAULT_MAX_CPU_PCT * 100

    def to_dict(self) -> dict:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_total_mb": round(self.ram_total_mb, 0),
            "ram_used_mb": round(self.ram_used_mb, 0),
            "ram_available_mb": round(self.ram_available_mb, 0),
            "ram_usage_pct": round(self.ram_usage_pct * 100, 1),
            "ram_headroom_mb": round(self.ram_headroom_mb, 0),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "max_workers_by_ram": self.max_workers_by_ram,
            "max_workers_by_cpu": self.max_workers_by_cpu,
            "is_ram_constrained": self.is_ram_constrained,
            "is_cpu_constrained": self.is_cpu_constrained,
            "cpu_count": self.cpu_count,
        }


def get_resource_snapshot() -> ResourceSnapshot:
    """Take a point-in-time snapshot of system resources.

    Uses psutil if available, falls back to /proc on Linux.
    """
    cpu_percent = 0.0
    ram_total_mb = 0.0
    ram_available_mb = 0.0
    ram_used_mb = 0.0
    disk_free_gb = 0.0
    load_1m = 0.0
    load_5m = 0.0
    cpu_count = os.cpu_count() or 1

    # --- RAM ---
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_total_mb = mem.total / (1024**2)
        ram_available_mb = mem.available / (1024**2)
        ram_used_mb = ram_total_mb - ram_available_mb
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            ram_total_mb = total_kb / 1024
            ram_available_mb = avail_kb / 1024
            ram_used_mb = ram_total_mb - ram_available_mb
        except (FileNotFoundError, OSError, ValueError):
            pass

    ram_usage_pct = ram_used_mb / ram_total_mb if ram_total_mb > 0 else 0.0

    # --- CPU ---
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except ImportError:
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
                load_1m = float(parts[0])
                load_5m = float(parts[1])
            cpu_percent = min(100.0, (load_1m / cpu_count) * 100.0)
        except (FileNotFoundError, OSError, ValueError, IndexError):
            pass

    # --- Disk ---
    try:
        stat = os.statvfs("/")
        disk_free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    except (OSError, ValueError):
        pass

    return ResourceSnapshot(
        cpu_percent=cpu_percent,
        ram_total_mb=ram_total_mb,
        ram_used_mb=ram_used_mb,
        ram_available_mb=ram_available_mb,
        ram_usage_pct=ram_usage_pct,
        disk_free_gb=disk_free_gb,
        load_avg_1m=load_1m,
        load_avg_5m=load_5m,
        cpu_count=cpu_count,
        timestamp=time.time(),
    )


def compute_optimal_workers(
    requested: int,
    snapshot: Optional[ResourceSnapshot] = None,
    *,
    memory_per_worker_mb: float = DEFAULT_MEMORY_PER_WORKER_MB,
    ram_reserve_mb: float = DEFAULT_RAM_RESERVE_MB,
    max_ram_usage_pct: float = DEFAULT_MAX_RAM_USAGE_PCT,
) -> int:
    """Compute the optimal number of workers given resource constraints.

    Returns the minimum of:
    1. Requested worker count
    2. RAM-based limit (available RAM / memory per worker)
    3. CPU-based limit (based on current CPU usage)
    4. CPU count

    Args:
        requested: Number of workers the caller wants.
        snapshot: Pre-computed resource snapshot (takes a new one if None).
        memory_per_worker_mb: Estimated RAM per worker.
        ram_reserve_mb: RAM to reserve for OS.
        max_ram_usage_pct: Maximum fraction of RAM to use.

    Returns:
        Optimal worker count (at least 1).
    """
    if snapshot is None:
        snapshot = get_resource_snapshot()

    cpu_count = snapshot.cpu_count

    # RAM-based limit
    available_mb = snapshot.ram_available_mb - ram_reserve_mb
    if available_mb > 0 and memory_per_worker_mb > 0:
        ram_limit = max(1, int(available_mb / memory_per_worker_mb))
    else:
        ram_limit = cpu_count  # Can't measure RAM, fall back to CPU

    # CPU-based limit
    cpu_headroom = max(0.1, 1.0 - (snapshot.cpu_percent / 100.0))
    cpu_limit = max(1, int(cpu_count * cpu_headroom))

    # Don't exceed RAM usage cap
    if snapshot.ram_total_mb > 0:
        max_ram_mb = snapshot.ram_total_mb * max_ram_usage_pct
        current_used = snapshot.ram_used_mb
        ram_budget_mb = max(0, max_ram_mb - current_used)
        if memory_per_worker_mb > 0:
            ram_budget_limit = max(1, int(ram_budget_mb / memory_per_worker_mb))
        else:
            ram_budget_limit = cpu_count
    else:
        ram_budget_limit = cpu_count

    optimal = min(requested, cpu_count, ram_limit, cpu_limit, ram_budget_limit)
    optimal = max(1, optimal)

    if optimal < requested:
        reason_parts = []
        if optimal == cpu_limit:
            reason_parts.append(f"CPU {snapshot.cpu_percent:.0f}%")
        if optimal == ram_limit:
            reason_parts.append(f"RAM {snapshot.ram_available_mb:.0f}MB free")
        if optimal == ram_budget_limit:
            reason_parts.append(f"RAM cap {max_ram_usage_pct:.0%}")
        logger.info(
            "Throttled workers: %d → %d (%s)",
            requested, optimal, ", ".join(reason_parts),
        )

    return optimal


class ResourceMonitor:
    """Periodic resource monitoring for adaptive throttling.

    Usage:
        monitor = ResourceMonitor()
        with monitor:
            workers = monitor.get_workers(8)
            # ... use workers ...
            workers = monitor.get_workers(8)  # May return fewer if resources dropped
    """

    def __init__(
        self,
        check_interval: float = DEFAULT_CHECK_INTERVAL_S,
        memory_per_worker_mb: float = DEFAULT_MEMORY_PER_WORKER_MB,
        min_workers: int = 1,
    ):
        self.check_interval = check_interval
        self.memory_per_worker_mb = memory_per_worker_mb
        self.min_workers = min_workers
        self._last_check: float = 0.0
        self._cached_snapshot: Optional[ResourceSnapshot] = None
        self._last_recommendation: int = 0
        self._check_count: int = 0
        self._throttle_count: int = 0

    def check(self) -> ResourceSnapshot:
        """Take a fresh resource snapshot."""
        self._cached_snapshot = get_resource_snapshot()
        self._last_check = time.time()
        self._check_count += 1
        return self._cached_snapshot

    def get_snapshot(self) -> ResourceSnapshot:
        """Get cached snapshot, refreshing if stale."""
        if (
            self._cached_snapshot is None
            or (time.time() - self._last_check) > self.check_interval
        ):
            return self.check()
        return self._cached_snapshot

    def get_workers(self, requested: int) -> int:
        """Get optimal worker count, potentially throttled from requested.

        Caches the check result for check_interval seconds to avoid
        excessive syscalls.
        """
        snapshot = self.get_snapshot()
        optimal = compute_optimal_workers(
            requested,
            snapshot,
            memory_per_worker_mb=self.memory_per_worker_mb,
        )
        optimal = max(self.min_workers, optimal)

        if optimal < requested:
            self._throttle_count += 1
            logger.debug(
                "ResourceMonitor: throttled %d → %d (check #%d, throttle #%d)",
                requested, optimal, self._check_count, self._throttle_count,
            )

        self._last_recommendation = optimal
        return optimal

    def get_status(self) -> dict:
        """Get monitor status for logging/dashboard."""
        snapshot = self.get_snapshot()
        return {
            "checks": self._check_count,
            "throttles": self._throttle_count,
            "last_recommendation": self._last_recommendation,
            "snapshot": snapshot.to_dict(),
        }

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, *args):
        pass
