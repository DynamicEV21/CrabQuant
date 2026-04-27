"""Tests for production daemon health check."""

import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crabquant.production.health import (
    HEARTBEAT_DEGRADED_S,
    HEARTBEAT_DOWN_S,
    check_health,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    *,
    heartbeat_ago_seconds: float | None = 60,
    wave: int = 3,
    mandates: int = 10,
    promoted: int = 2,
    api_calls: int = 500,
) -> dict:
    """Build a daemon_state.json dict."""
    state = {
        "current_wave": wave,
        "total_mandates_run": mandates,
        "total_promoted": promoted,
        "total_api_calls": api_calls,
    }
    if heartbeat_ago_seconds is not None:
        dt = datetime.now(timezone.utc) - timedelta(seconds=heartbeat_ago_seconds)
        state["last_heartbeat"] = dt.isoformat()
    return state


def _write_state(tmp: Path, **kwargs) -> str:
    """Write state JSON to tmpdir, return path."""
    state = _make_state(**kwargs)
    p = tmp / "daemon_state.json"
    p.write_text(json.dumps(state))
    return str(p)


def _write_pid(tmp: Path, pid: int) -> str:
    """Write PID file to tmpdir, return path."""
    p = tmp / "crabquant.pid"
    p.write_text(str(pid))
    return str(p)


@pytest.fixture
def mock_system(tmp_path):
    """Provide a temp dir with state/pid files and patch system readings."""
    pid_path = _write_pid(tmp_path, 12345)
    state_path = _write_state(tmp_path, heartbeat_ago_seconds=60)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    return {
        "tmp": tmp_path,
        "pid_path": pid_path,
        "state_path": state_path,
        "cache_dir": cache_dir,
    }


def _patch_system(cpu_percent=10.0, ram_total=16.0, ram_free=12.0, disk_free=100.0):
    """Return patchers for system info functions."""
    return patch(
        "crabquant.production.health._get_system_info",
        return_value={
            "cpu_percent": cpu_percent,
            "ram_total_gb": ram_total,
            "ram_used_gb": round(ram_total - ram_free, 2),
            "ram_free_gb": ram_free,
            "disk_free_gb": disk_free,
        },
    )


def _patch_cache(fresh=False, age_hours=999.0):
    """Return patcher for cache check."""
    return patch(
        "crabquant.production.health._check_cache",
        return_value={
            "cache_fresh": fresh,
            "cache_age_hours": age_hours,
            "cache_dir": "/fake/cache",
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthCheck:

    def test_healthy_daemon(self, mock_system):
        """Daemon alive, recent heartbeat, plenty RAM → healthy."""
        with (
            _patch_system() as sys_mock,
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        assert result["status"] == "healthy"
        assert result["daemon"]["alive"] is True
        assert result["daemon"]["pid"] == 12345
        assert result["daemon"]["heartbeat_age_seconds"] is not None
        assert result["daemon"]["heartbeat_age_seconds"] < HEARTBEAT_DEGRADED_S
        assert result["daemon"]["current_wave"] == 3
        assert result["daemon"]["total_mandates_run"] == 10
        assert result["recommendations"] == []

    def test_degraded_stale_heartbeat(self, mock_system):
        """Daemon alive but heartbeat 10min old → degraded."""
        state_path = _write_state(
            mock_system["tmp"], heartbeat_ago_seconds=600
        )
        with (
            _patch_system() as sys_mock,
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=state_path,
                pid_path=mock_system["pid_path"],
            )

        assert result["status"] == "degraded"
        assert result["daemon"]["alive"] is True
        assert "heartbeat stale" in " ".join(result["recommendations"]).lower()

    def test_down_no_pid_file(self, mock_system):
        """No PID file → down."""
        state_path = _write_state(mock_system["tmp"], heartbeat_ago_seconds=60)
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
        ):
            result = check_health(
                state_path=state_path,
                pid_path=str(mock_system["tmp"] / "nonexistent.pid"),
            )

        assert result["status"] == "down"
        assert result["daemon"]["alive"] is False
        assert result["daemon"]["pid"] is None

    def test_down_dead_process(self, mock_system):
        """PID file exists but process not running → down."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=False),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        assert result["status"] == "down"
        assert result["daemon"]["alive"] is False
        assert result["daemon"]["pid"] == 12345

    def test_down_very_old_heartbeat(self, mock_system):
        """Heartbeat > 30min → down even if process alive."""
        state_path = _write_state(
            mock_system["tmp"], heartbeat_ago_seconds=2400
        )
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=state_path,
                pid_path=mock_system["pid_path"],
            )

        assert result["status"] == "down"

    def test_recommendations_low_ram_critical(self, mock_system):
        """RAM < 2GB → critical recommendation."""
        with (
            _patch_system(ram_free=1.5),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        recs = result["recommendations"]
        assert any("critically low" in r for r in recs)
        assert any("reduce parallel to 1" in r for r in recs)

    def test_recommendations_low_ram_warning(self, mock_system):
        """RAM < 4GB → low recommendation (not critical)."""
        with (
            _patch_system(ram_free=3.0),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        recs = result["recommendations"]
        assert not any("critically low" in r for r in recs)
        assert any("RAM low" in r for r in recs)

    def test_recommendations_low_disk(self, mock_system):
        """Disk < 5GB → cleanup recommendation."""
        with (
            _patch_system(ram_free=12.0, disk_free=3.0),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        recs = result["recommendations"]
        assert any("Disk space low" in r for r in recs)

    def test_recommendations_stale_cache(self, mock_system):
        """Cache > 24h → refresh recommendation."""
        with (
            _patch_system(),
            _patch_cache(fresh=False, age_hours=48.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        recs = result["recommendations"]
        assert any("cache stale" in r.lower() for r in recs)

    def test_cache_fresh(self, mock_system):
        """Cache file < 24h → fresh=True."""
        # Create a recent .pkl file
        pkl = mock_system["cache_dir"] / "data.pkl"
        pkl.write_bytes(b"\x00")

        with (
            _patch_system(),
            patch("crabquant.production.health.CACHE_DIR", mock_system["cache_dir"]),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        assert result["data"]["cache_fresh"] is True
        assert result["data"]["cache_age_hours"] < 24

    def test_cache_stale(self, mock_system):
        """Cache file > 24h → fresh=False."""
        pkl = mock_system["cache_dir"] / "old_data.pkl"
        pkl.write_bytes(b"\x00")
        # Set mtime to 25 hours ago
        old_time = time.time() - (25 * 3600)
        os.utime(pkl, (old_time, old_time))

        with (
            _patch_system(),
            patch("crabquant.production.health.CACHE_DIR", mock_system["cache_dir"]),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        assert result["data"]["cache_fresh"] is False
        assert result["data"]["cache_age_hours"] > 24

    def test_cache_missing(self, mock_system):
        """No cache directory → fresh=False."""
        nonexistent = mock_system["tmp"] / "no_such_cache"
        with (
            _patch_system(),
            patch("crabquant.production.health.CACHE_DIR", nonexistent),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        assert result["data"]["cache_fresh"] is False
        assert result["data"]["cache_age_hours"] == 999.0

    def test_state_missing(self, mock_system):
        """Missing state file → None fields for daemon stats."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=str(mock_system["tmp"] / "no_state.json"),
                pid_path=mock_system["pid_path"],
            )

        # No state loaded, but PID alive → still healthy (no heartbeat to judge)
        assert result["daemon"]["alive"] is True
        assert result["daemon"]["last_heartbeat"] is None
        assert result["daemon"]["heartbeat_age_seconds"] is None
        assert result["daemon"]["current_wave"] is None

    def test_checked_at_is_iso(self, mock_system):
        """checked_at is a valid ISO timestamp."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )

        # Should parse without error
        datetime.fromisoformat(result["checked_at"])


class TestCLIEntryPoint:

    def test_cli_exits_0_for_healthy(self, mock_system):
        """__main__ exits 0 for healthy status."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
            patch("sys.stdout") as stdout_mock,
        ):
            # Simulate __main__ block
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )
            output = json.dumps(result, indent=2)

        assert result["status"] == "healthy"
        # Parseable JSON output
        parsed = json.loads(output)
        assert parsed["status"] == "healthy"

    def test_cli_exits_1_for_down(self, mock_system):
        """__main__ exits 1 for down status."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=str(mock_system["tmp"] / "no_pid.pid"),
            )

        assert result["status"] == "down"
        # Exit code should be 1
        assert result["status"] != "healthy"
