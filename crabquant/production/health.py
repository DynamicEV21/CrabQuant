"""
Production Daemon Health Check

Monitors daemon process, system resources, and data cache freshness.
Designed for quick diagnostic checks and monitoring integration.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "crabquant" / "data" / "cache"

# Thresholds
HEARTBEAT_DEGRADED_S = 300   # 5 minutes
HEARTBEAT_DOWN_S = 1800      # 30 minutes
CACHE_FRESH_HOURS = 24
RAM_CRITICAL_GB = 2.0
RAM_LOW_GB = 4.0
DISK_LOW_GB = 5.0


def _read_pid(pid_path: str) -> int | None:
    """Read PID from file, return None if missing or invalid."""
    try:
        with open(pid_path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _process_alive(pid: int) -> bool:
    """Check if a process is alive by sending signal 0."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _load_state(state_path: str) -> dict | None:
    """Load daemon state from JSON file."""
    try:
        with open(state_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _heartbeat_age(state: dict) -> float | None:
    """Compute seconds since last heartbeat from state dict."""
    hb = state.get("last_heartbeat")
    if not hb:
        return None
    try:
        # Support ISO format with or without Z
        if hb.endswith("Z"):
            hb = hb[:-1] + "+00:00"
        dt = datetime.fromisoformat(hb)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except (ValueError, TypeError):
        return None


def _get_system_info() -> dict:
    """Get CPU, RAM, and disk info. Tries psutil first, falls back to /proc."""
    cpu_percent = 0.0
    ram_total_gb = 0.0
    ram_used_gb = 0.0
    ram_free_gb = 0.0
    disk_free_gb = 0.0

    # --- RAM ---
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_total_gb = mem.total / (1024**3)
        ram_free_gb = mem.available / (1024**3)
        ram_used_gb = ram_total_gb - ram_free_gb
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
            ram_total_gb = total_kb / (1024**2)
            ram_free_gb = avail_kb / (1024**2)
            ram_used_gb = ram_total_gb - ram_free_gb
        except (FileNotFoundError, OSError, ValueError):
            pass

    # --- CPU ---
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except ImportError:
        try:
            with open("/proc/loadavg") as f:
                load1, _, _ = f.read().split()[:3]
            cpu_count = os.cpu_count() or 1
            cpu_percent = (float(load1) / cpu_count) * 100.0
        except (FileNotFoundError, OSError, ValueError):
            pass

    # --- Disk ---
    try:
        stat = os.statvfs(str(BASE_DIR))
        disk_free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    except (OSError, ValueError):
        pass

    return {
        "cpu_percent": round(cpu_percent, 1),
        "ram_total_gb": round(ram_total_gb, 2),
        "ram_used_gb": round(ram_used_gb, 2),
        "ram_free_gb": round(ram_free_gb, 2),
        "disk_free_gb": round(disk_free_gb, 2),
    }


def _check_cache() -> dict:
    """Check price data cache freshness."""
    cache_dir = str(CACHE_DIR)

    if not CACHE_DIR.exists() or not CACHE_DIR.is_dir():
        return {"cache_fresh": False, "cache_age_hours": 999.0, "cache_dir": cache_dir}

    pkl_files = list(CACHE_DIR.glob("*.pkl"))
    if not pkl_files:
        return {"cache_fresh": False, "cache_age_hours": 999.0, "cache_dir": cache_dir}

    newest = max(pkl_files, key=lambda p: p.stat().st_mtime)
    age_seconds = time.time() - newest.stat().st_mtime
    age_hours = age_seconds / 3600.0

    return {
        "cache_fresh": age_hours < CACHE_FRESH_HOURS,
        "cache_age_hours": round(age_hours, 1),
        "cache_dir": cache_dir,
    }


def _build_recommendations(
    ram_free_gb: float,
    disk_free_gb: float,
    heartbeat_age_s: float | None,
    cache_fresh: bool,
) -> list[str]:
    """Generate actionable recommendations based on system state."""
    recs: list[str] = []

    if ram_free_gb < RAM_CRITICAL_GB:
        recs.append("RAM critically low — reduce parallel to 1")
    elif ram_free_gb < RAM_LOW_GB:
        recs.append("RAM low — consider reducing parallel to 2")

    if disk_free_gb < DISK_LOW_GB:
        recs.append("Disk space low — clean up refinement_runs")

    if heartbeat_age_s is not None and heartbeat_age_s > HEARTBEAT_DEGRADED_S:
        recs.append("Daemon heartbeat stale — may need restart")

    if not cache_fresh:
        recs.append("Price data cache stale — consider refreshing")

    return recs


def check_health(
    state_path: str = "results/daemon_state.json",
    pid_path: str = "crabquant.pid",
) -> dict:
    """
    Check health of the CrabQuant daemon.

    Returns a dict with status, daemon info, system metrics,
    cache state, and recommendations.
    """
    checked_at = datetime.now(timezone.utc).isoformat()

    # --- Daemon check ---
    pid = _read_pid(pid_path)
    alive = False
    if pid is not None:
        alive = _process_alive(pid)

    state = _load_state(state_path)
    hb_age = _heartbeat_age(state) if state else None

    # Determine overall status
    if pid is None or not alive:
        status = "down"
    elif hb_age is not None and hb_age > HEARTBEAT_DOWN_S:
        status = "down"
    elif hb_age is not None and hb_age > HEARTBEAT_DEGRADED_S:
        status = "degraded"
    else:
        status = "healthy"

    daemon = {
        "alive": alive,
        "pid": pid,
        "last_heartbeat": state.get("last_heartbeat") if state else None,
        "heartbeat_age_seconds": round(hb_age, 1) if hb_age is not None else None,
        "current_wave": state.get("current_wave") if state else None,
        "total_mandates_run": state.get("total_mandates_run") if state else None,
        "total_promoted": state.get("total_strategies_promoted") if state else None,
        "total_api_calls": state.get("total_api_calls") if state else None,
    }

    # --- System check ---
    system = _get_system_info()

    # --- Cache check ---
    data = _check_cache()

    # --- Recommendations ---
    recommendations = _build_recommendations(
        ram_free_gb=system["ram_free_gb"],
        disk_free_gb=system["disk_free_gb"],
        heartbeat_age_s=hb_age,
        cache_fresh=data["cache_fresh"],
    )

    return {
        "status": status,
        "daemon": daemon,
        "system": system,
        "data": data,
        "recommendations": recommendations,
        "checked_at": checked_at,
    }


if __name__ == "__main__":
    import sys

    result = check_health()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "healthy" else 1)
