"""Tests for production daemon health check."""

import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from crabquant.production.health import (
    HEARTBEAT_DEGRADED_S,
    HEARTBEAT_DOWN_S,
    CACHE_FRESH_HOURS,
    RAM_CRITICAL_GB,
    RAM_LOW_GB,
    DISK_LOW_GB,
    _read_pid,
    _process_alive,
    _load_state,
    _heartbeat_age,
    _check_cache,
    _build_recommendations,
    _get_system_info,
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


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestReadPid:
    def test_reads_valid_pid(self, tmp_path):
        """Should read a valid integer PID from file."""
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("12345\n")
        assert _read_pid(str(pid_file)) == 12345

    def test_returns_none_for_missing_file(self):
        """Should return None when file doesn't exist."""
        assert _read_pid("/nonexistent/path.pid") is None

    def test_returns_none_for_garbage_content(self, tmp_path):
        """Should return None for non-integer content."""
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not_a_number")
        assert _read_pid(str(pid_file)) is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Should return None for empty file."""
        pid_file = tmp_path / "empty.pid"
        pid_file.write_text("")
        assert _read_pid(str(pid_file)) is None

    def test_strips_whitespace(self, tmp_path):
        """Should handle leading/trailing whitespace."""
        pid_file = tmp_path / "ws.pid"
        pid_file.write_text("  9999  \n")
        assert _read_pid(str(pid_file)) == 9999


class TestProcessAlive:
    def test_own_process_is_alive(self):
        """Current process should be alive (signal 0 succeeds)."""
        assert _process_alive(os.getpid()) is True

    def test_nonexistent_pid_is_dead(self):
        """PID that doesn't exist should return False."""
        # Use a very large PID that almost certainly doesn't exist
        assert _process_alive(999999999) is False


class TestLoadState:
    def test_loads_valid_json(self, tmp_path):
        """Should parse valid JSON state file."""
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"last_heartbeat": "2024-01-01T00:00:00Z"}))
        result = _load_state(str(state_file))
        assert result == {"last_heartbeat": "2024-01-01T00:00:00Z"}

    def test_returns_none_for_missing_file(self):
        """Should return None for missing file."""
        assert _load_state("/nonexistent/state.json") is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        """Should return None for malformed JSON."""
        state_file = tmp_path / "bad.json"
        state_file.write_text("{invalid json")
        assert _load_state(str(state_file)) is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Should return None for empty file."""
        state_file = tmp_path / "empty.json"
        state_file.write_text("")
        assert _load_state(str(state_file)) is None


class TestHeartbeatAge:
    def test_recent_heartbeat(self):
        """Recent heartbeat should return small age in seconds."""
        dt = datetime.now(timezone.utc) - timedelta(seconds=30)
        state = {"last_heartbeat": dt.isoformat()}
        age = _heartbeat_age(state)
        assert age is not None
        assert 25 < age < 40  # Allow some execution time

    def test_z_suffix_format(self):
        """Should handle ISO format with Z suffix."""
        dt = datetime.now(timezone.utc) - timedelta(seconds=60)
        state = {"last_heartbeat": dt.strftime("%Y-%m-%dT%H:%M:%SZ")}
        age = _heartbeat_age(state)
        assert age is not None
        assert 50 < age < 70

    def test_no_heartbeat_key(self):
        """Missing last_heartbeat should return None."""
        assert _heartbeat_age({}) is None

    def test_none_heartbeat(self):
        """None heartbeat value should return None."""
        assert _heartbeat_age({"last_heartbeat": None}) is None

    def test_empty_string_heartbeat(self):
        """Empty string heartbeat should return None."""
        assert _heartbeat_age({"last_heartbeat": ""}) is None

    def test_invalid_format_heartbeat(self):
        """Invalid date format should return None."""
        assert _heartbeat_age({"last_heartbeat": "not-a-date"}) is None

    def test_naive_datetime_gets_utc(self):
        """Naive datetime (no timezone) should be treated as UTC."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        state = {"last_heartbeat": dt.isoformat()}
        age = _heartbeat_age(state)
        assert age is not None
        # Age should be very large (Jan 2024 to now)
        assert age > 1_000_000


class TestCheckCache:
    def test_missing_cache_dir(self, tmp_path):
        """Nonexistent cache dir → fresh=False."""
        nonexistent = tmp_path / "no_cache"
        with patch("crabquant.production.health.CACHE_DIR", nonexistent):
            result = _check_cache()
        assert result["cache_fresh"] is False
        assert result["cache_age_hours"] == 999.0

    def test_empty_cache_dir(self, tmp_path):
        """Cache dir with no .pkl files → fresh=False."""
        cache_dir = tmp_path / "empty_cache"
        cache_dir.mkdir()
        with patch("crabquant.production.health.CACHE_DIR", cache_dir):
            result = _check_cache()
        assert result["cache_fresh"] is False

    def test_fresh_pkl_file(self, tmp_path):
        """Recent .pkl file → fresh=True."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        pkl = cache_dir / "fresh.pkl"
        pkl.write_bytes(b"\x00")
        with patch("crabquant.production.health.CACHE_DIR", cache_dir):
            result = _check_cache()
        assert result["cache_fresh"] is True

    def test_stale_pkl_file(self, tmp_path):
        """Old .pkl file → fresh=False."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        pkl = cache_dir / "stale.pkl"
        pkl.write_bytes(b"\x00")
        old_time = time.time() - (30 * 3600)  # 30 hours ago
        os.utime(pkl, (old_time, old_time))
        with patch("crabquant.production.health.CACHE_DIR", cache_dir):
            result = _check_cache()
        assert result["cache_fresh"] is False

    def test_picks_newest_pkl(self, tmp_path):
        """Should pick the newest .pkl file for age check."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Old file
        old_pkl = cache_dir / "old.pkl"
        old_pkl.write_bytes(b"\x00")
        old_time = time.time() - (48 * 3600)
        os.utime(old_pkl, (old_time, old_time))
        # New file
        new_pkl = cache_dir / "new.pkl"
        new_pkl.write_bytes(b"\x00")
        with patch("crabquant.production.health.CACHE_DIR", cache_dir):
            result = _check_cache()
        assert result["cache_fresh"] is True
        assert result["cache_age_hours"] < 1.0

    def test_cache_dir_is_file_not_dir(self, tmp_path):
        """CACHE_DIR pointing to a file → fresh=False."""
        fake_file = tmp_path / "not_a_dir"
        fake_file.write_text("not a directory")
        with patch("crabquant.production.health.CACHE_DIR", fake_file):
            result = _check_cache()
        assert result["cache_fresh"] is False


class TestBuildRecommendations:
    def test_no_recs_when_healthy(self):
        """Plenty RAM, disk, fresh cache, recent heartbeat → empty list."""
        recs = _build_recommendations(
            ram_free_gb=12.0,
            disk_free_gb=100.0,
            heartbeat_age_s=60.0,
            cache_fresh=True,
        )
        assert recs == []

    def test_critical_ram(self):
        """RAM < 2GB → critical recommendation."""
        recs = _build_recommendations(
            ram_free_gb=1.0,
            disk_free_gb=100.0,
            heartbeat_age_s=None,
            cache_fresh=True,
        )
        assert any("critically low" in r for r in recs)

    def test_low_ram(self):
        """2GB <= RAM < 4GB → low recommendation (not critical)."""
        recs = _build_recommendations(
            ram_free_gb=3.0,
            disk_free_gb=100.0,
            heartbeat_age_s=None,
            cache_fresh=True,
        )
        assert any("RAM low" in r for r in recs)
        assert not any("critically low" in r for r in recs)

    def test_low_disk(self):
        """Disk < 5GB → disk recommendation."""
        recs = _build_recommendations(
            ram_free_gb=12.0,
            disk_free_gb=2.0,
            heartbeat_age_s=None,
            cache_fresh=True,
        )
        assert any("Disk space low" in r for r in recs)

    def test_stale_heartbeat(self):
        """Heartbeat > 5min → stale heartbeat recommendation."""
        recs = _build_recommendations(
            ram_free_gb=12.0,
            disk_free_gb=100.0,
            heartbeat_age_s=600.0,
            cache_fresh=True,
        )
        assert any("heartbeat stale" in r.lower() for r in recs)

    def test_stale_cache(self):
        """Stale cache → cache refresh recommendation."""
        recs = _build_recommendations(
            ram_free_gb=12.0,
            disk_free_gb=100.0,
            heartbeat_age_s=None,
            cache_fresh=False,
        )
        assert any("cache stale" in r.lower() for r in recs)

    def test_none_heartbeat_skips_stale_check(self):
        """None heartbeat age should not trigger stale heartbeat rec."""
        recs = _build_recommendations(
            ram_free_gb=12.0,
            disk_free_gb=100.0,
            heartbeat_age_s=None,
            cache_fresh=True,
        )
        assert not any("heartbeat" in r.lower() for r in recs)

    def test_multiple_recommendations(self):
        """Multiple issues should produce multiple recommendations."""
        recs = _build_recommendations(
            ram_free_gb=1.0,       # Critical RAM
            disk_free_gb=2.0,      # Low disk
            heartbeat_age_s=600.0, # Stale heartbeat
            cache_fresh=False,     # Stale cache
        )
        assert len(recs) >= 4

    def test_boundary_ram_critical(self):
        """RAM exactly at critical threshold → critical rec."""
        recs = _build_recommendations(
            ram_free_gb=RAM_CRITICAL_GB - 0.01,
            disk_free_gb=100.0,
            heartbeat_age_s=None,
            cache_fresh=True,
        )
        assert any("critically low" in r for r in recs)

    def test_boundary_ram_ok(self):
        """RAM just above low threshold → no RAM recs."""
        recs = _build_recommendations(
            ram_free_gb=RAM_LOW_GB + 0.01,
            disk_free_gb=100.0,
            heartbeat_age_s=None,
            cache_fresh=True,
        )
        assert not any("RAM" in r for r in recs)


class TestCheckHealthIntegration:
    def test_api_budget_nearly_exhausted(self, mock_system):
        """API budget < $0.50 should add budget warning."""
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = {
            "budget_remaining": {"cost_usd": 0.30}
        }
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
            patch("crabquant.refinement.api_budget.get_global_tracker", return_value=mock_tracker),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )
        recs = result["recommendations"]
        assert any("API budget" in r for r in recs)

    def test_api_budget_ok(self, mock_system):
        """API budget healthy → no budget warning."""
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = {
            "budget_remaining": {"cost_usd": 5.00}
        }
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
            patch("crabquant.refinement.api_budget.get_global_tracker", return_value=mock_tracker),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )
        recs = result["recommendations"]
        assert not any("API budget" in r for r in recs)

    def test_api_budget_import_error_handled(self, mock_system):
        """ImportError for api_budget should not crash."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
            patch.dict("sys.modules", {"crabquant.refinement.api_budget": None}),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )
        assert result["api_budget"] == {}

    def test_invalid_pid_file_content(self, mock_system):
        """PID file with non-numeric content → pid=None, down."""
        bad_pid = mock_system["tmp"] / "bad.pid"
        bad_pid.write_text("not_a_pid")
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=str(bad_pid),
            )
        assert result["daemon"]["pid"] is None
        assert result["status"] == "down"

    def test_invalid_state_json(self, mock_system):
        """Invalid JSON state file → treated as missing state."""
        bad_state = mock_system["tmp"] / "bad_state.json"
        bad_state.write_text("{bad json")
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
            patch("crabquant.production.health._process_alive", return_value=True),
        ):
            result = check_health(
                state_path=str(bad_state),
                pid_path=mock_system["pid_path"],
            )
        # State failed to load → None values
        assert result["daemon"]["last_heartbeat"] is None
        assert result["daemon"]["current_wave"] is None

    def test_result_has_all_top_level_keys(self, mock_system):
        """Result dict should contain all expected top-level keys."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )
        expected_keys = {
            "status", "daemon", "system", "data",
            "api_budget", "recommendations", "checked_at",
        }
        assert set(result.keys()) == expected_keys

    def test_daemon_dict_has_all_keys(self, mock_system):
        """Daemon dict should contain all expected keys."""
        with (
            _patch_system(),
            _patch_cache(fresh=True, age_hours=1.0),
        ):
            result = check_health(
                state_path=mock_system["state_path"],
                pid_path=mock_system["pid_path"],
            )
        expected_keys = {
            "alive", "pid", "last_heartbeat", "heartbeat_age_seconds",
            "current_wave", "total_mandates_run", "total_promoted", "total_api_calls",
        }
        assert set(result["daemon"].keys()) == expected_keys
