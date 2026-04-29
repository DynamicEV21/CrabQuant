"""Tests for crabquant.production.health"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crabquant.production.health import (
    _read_pid,
    _process_alive,
    _load_state,
    _heartbeat_age,
    _build_recommendations,
    _check_cache,
    check_health,
    HEARTBEAT_DEGRADED_S,
    HEARTBEAT_DOWN_S,
    RAM_CRITICAL_GB,
    RAM_LOW_GB,
    DISK_LOW_GB,
)


class TestReadPid:
    def test_valid_pid(self, tmp_path):
        pid_file = tmp_path / "crabquant.pid"
        pid_file.write_text("12345\n")
        assert _read_pid(str(pid_file)) == 12345

    def test_missing_file(self):
        assert _read_pid("/nonexistent/crabquant.pid") is None

    def test_invalid_content(self, tmp_path):
        pid_file = tmp_path / "crabquant.pid"
        pid_file.write_text("not_a_number")
        assert _read_pid(str(pid_file)) is None

    def test_empty_file(self, tmp_path):
        pid_file = tmp_path / "crabquant.pid"
        pid_file.write_text("")
        assert _read_pid(str(pid_file)) is None


class TestProcessAlive:
    def test_own_pid_is_alive(self):
        # Our own process should be alive
        assert _process_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        # Use a PID that definitely doesn't exist
        assert _process_alive(9999999) is False


class TestLoadState:
    def test_valid_state(self, tmp_path):
        state = {"last_heartbeat": "2026-01-01T00:00:00Z", "current_wave": 3}
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state))
        loaded = _load_state(str(state_file))
        assert loaded["current_wave"] == 3

    def test_missing_file(self):
        assert _load_state("/nonexistent/state.json") is None

    def test_invalid_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not json")
        assert _load_state(str(state_file)) is None


class TestHeartbeatAge:
    def test_recent_heartbeat(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=10)
        state = {"last_heartbeat": recent.isoformat()}
        age = _heartbeat_age(state)
        assert age is not None
        assert 0 <= age < 20  # Allow some slack

    def test_stale_heartbeat(self):
        stale = datetime.now(timezone.utc) - timedelta(minutes=10)
        state = {"last_heartbeat": stale.isoformat()}
        age = _heartbeat_age(state)
        assert age is not None
        assert age > 500  # > 8 minutes

    def test_no_heartbeat(self):
        state = {}
        assert _heartbeat_age(state) is None

    def test_z_suffix(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=5)
        state = {"last_heartbeat": recent.strftime("%Y-%m-%dT%H:%M:%SZ")}
        age = _heartbeat_age(state)
        assert age is not None
        assert age < 15

    def test_invalid_format(self):
        state = {"last_heartbeat": "not-a-date"}
        assert _heartbeat_age(state) is None


class TestBuildRecommendations:
    def test_all_healthy(self):
        recs = _build_recommendations(
            ram_free_gb=16.0, disk_free_gb=100.0,
            heartbeat_age_s=30.0, cache_fresh=True,
        )
        assert recs == []

    def test_critical_ram(self):
        recs = _build_recommendations(
            ram_free_gb=1.0, disk_free_gb=100.0,
            heartbeat_age_s=30.0, cache_fresh=True,
        )
        assert any("RAM critically low" in r for r in recs)

    def test_low_ram(self):
        recs = _build_recommendations(
            ram_free_gb=3.0, disk_free_gb=100.0,
            heartbeat_age_s=30.0, cache_fresh=True,
        )
        assert any("RAM low" in r for r in recs)

    def test_low_disk(self):
        recs = _build_recommendations(
            ram_free_gb=16.0, disk_free_gb=3.0,
            heartbeat_age_s=30.0, cache_fresh=True,
        )
        assert any("Disk space low" in r for r in recs)

    def test_stale_heartbeat(self):
        recs = _build_recommendations(
            ram_free_gb=16.0, disk_free_gb=100.0,
            heartbeat_age_s=600.0, cache_fresh=True,
        )
        assert any("heartbeat stale" in r for r in recs)

    def test_stale_cache(self):
        recs = _build_recommendations(
            ram_free_gb=16.0, disk_free_gb=100.0,
            heartbeat_age_s=30.0, cache_fresh=False,
        )
        assert any("stale" in r for r in recs)

    def test_multiple_issues(self):
        recs = _build_recommendations(
            ram_free_gb=1.0, disk_free_gb=3.0,
            heartbeat_age_s=600.0, cache_fresh=False,
        )
        assert len(recs) >= 3


class TestCheckCache:
    def test_nonexistent_cache_dir(self, tmp_path):
        with patch("crabquant.production.health.CACHE_DIR", tmp_path / "nonexistent"):
            result = _check_cache()
            assert result["cache_fresh"] is False
            assert result["cache_age_hours"] == 999.0

    def test_empty_cache_dir(self, tmp_path):
        with patch("crabquant.production.health.CACHE_DIR", tmp_path / "cache"):
            result = _check_cache()
            assert result["cache_fresh"] is False

    def test_fresh_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create a recent .pkl file
        (cache_dir / "spy.pkl").touch()
        with patch("crabquant.production.health.CACHE_DIR", cache_dir):
            result = _check_cache()
            assert result["cache_fresh"] is True
            assert result["cache_age_hours"] < 1.0

    def test_stale_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create an old .pkl file
        old_file = cache_dir / "spy.pkl"
        old_file.touch()
        # Set mtime to 48 hours ago
        old_time = time.time() - (48 * 3600)
        os.utime(old_file, (old_time, old_time))
        with patch("crabquant.production.health.CACHE_DIR", cache_dir):
            result = _check_cache()
            assert result["cache_fresh"] is False
            assert result["cache_age_hours"] > 47.0


class TestCheckHealth:
    def test_daemon_down_no_pid(self, tmp_path):
        with patch("crabquant.production.health._read_pid", return_value=None), \
             patch("crabquant.production.health._get_system_info", return_value={
                 "cpu_percent": 10.0, "ram_total_gb": 16.0,
                 "ram_used_gb": 4.0, "ram_free_gb": 12.0, "disk_free_gb": 100.0,
             }), \
             patch("crabquant.production.health._check_cache", return_value={
                 "cache_fresh": True, "cache_age_hours": 1.0, "cache_dir": "/tmp/cache",
             }), \
             patch("crabquant.refinement.api_budget.get_global_tracker", side_effect=Exception):
            result = check_health()
            assert result["status"] == "down"
            assert result["daemon"]["alive"] is False
            assert result["daemon"]["pid"] is None
            assert result["checked_at"] is not None

    def test_daemon_alive_healthy(self, tmp_path):
        recent = datetime.now(timezone.utc) - timedelta(seconds=10)
        with patch("crabquant.production.health._read_pid", return_value=12345), \
             patch("crabquant.production.health._process_alive", return_value=True), \
             patch("crabquant.production.health._load_state", return_value={
                 "last_heartbeat": recent.isoformat(),
                 "current_wave": 5,
                 "total_mandates_run": 100,
                 "total_strategies_promoted": 10,
                 "total_api_calls": 500,
             }), \
             patch("crabquant.production.health._get_system_info", return_value={
                 "cpu_percent": 25.0, "ram_total_gb": 16.0,
                 "ram_used_gb": 8.0, "ram_free_gb": 8.0, "disk_free_gb": 50.0,
             }), \
             patch("crabquant.production.health._check_cache", return_value={
                 "cache_fresh": True, "cache_age_hours": 2.0, "cache_dir": "/tmp/cache",
             }), \
             patch("crabquant.refinement.api_budget.get_global_tracker", side_effect=Exception):
            result = check_health()
            assert result["status"] == "healthy"
            assert result["daemon"]["alive"] is True
            assert result["daemon"]["pid"] == 12345
            assert result["daemon"]["current_wave"] == 5
            assert result["daemon"]["total_mandates_run"] == 100

    def test_daemon_degraded_stale_heartbeat(self, tmp_path):
        stale = datetime.now(timezone.utc) - timedelta(minutes=10)
        with patch("crabquant.production.health._read_pid", return_value=12345), \
             patch("crabquant.production.health._process_alive", return_value=True), \
             patch("crabquant.production.health._load_state", return_value={
                 "last_heartbeat": stale.isoformat(),
             }), \
             patch("crabquant.production.health._get_system_info", return_value={
                 "cpu_percent": 10.0, "ram_total_gb": 16.0,
                 "ram_used_gb": 4.0, "ram_free_gb": 12.0, "disk_free_gb": 100.0,
             }), \
             patch("crabquant.production.health._check_cache", return_value={
                 "cache_fresh": True, "cache_age_hours": 1.0, "cache_dir": "/tmp/cache",
             }), \
             patch("crabquant.refinement.api_budget.get_global_tracker", side_effect=Exception):
            result = check_health()
            assert result["status"] == "degraded"

    def test_daemon_down_dead_process(self, tmp_path):
        with patch("crabquant.production.health._read_pid", return_value=12345), \
             patch("crabquant.production.health._process_alive", return_value=False), \
             patch("crabquant.production.health._load_state", return_value=None), \
             patch("crabquant.production.health._get_system_info", return_value={
                 "cpu_percent": 10.0, "ram_total_gb": 16.0,
                 "ram_used_gb": 4.0, "ram_free_gb": 12.0, "disk_free_gb": 100.0,
             }), \
             patch("crabquant.production.health._check_cache", return_value={
                 "cache_fresh": True, "cache_age_hours": 1.0, "cache_dir": "/tmp/cache",
             }), \
             patch("crabquant.refinement.api_budget.get_global_tracker", side_effect=Exception):
            result = check_health()
            assert result["status"] == "down"
            assert result["daemon"]["alive"] is False

    def test_api_budget_integration(self, tmp_path):
        recent = datetime.now(timezone.utc) - timedelta(seconds=10)
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = {
            "budget_remaining": {"cost_usd": 0.3},
        }
        with patch("crabquant.production.health._read_pid", return_value=None), \
             patch("crabquant.production.health._get_system_info", return_value={
                 "cpu_percent": 10.0, "ram_total_gb": 16.0,
                 "ram_used_gb": 4.0, "ram_free_gb": 12.0, "disk_free_gb": 100.0,
             }), \
             patch("crabquant.production.health._check_cache", return_value={
                 "cache_fresh": True, "cache_age_hours": 1.0, "cache_dir": "/tmp/cache",
             }), \
             patch("crabquant.refinement.api_budget.get_global_tracker", return_value=mock_tracker):
            result = check_health()
            assert result["api_budget"]["budget_remaining"]["cost_usd"] == 0.3
            assert any("API budget nearly exhausted" in r for r in result["recommendations"])

    def test_result_has_required_keys(self, tmp_path):
        with patch("crabquant.production.health._read_pid", return_value=None), \
             patch("crabquant.production.health._get_system_info", return_value={
                 "cpu_percent": 0.0, "ram_total_gb": 0.0,
                 "ram_used_gb": 0.0, "ram_free_gb": 0.0, "disk_free_gb": 0.0,
             }), \
             patch("crabquant.production.health._check_cache", return_value={
                 "cache_fresh": False, "cache_age_hours": 999.0, "cache_dir": "/tmp",
             }), \
             patch("crabquant.refinement.api_budget.get_global_tracker", side_effect=Exception):
            result = check_health()
            required_keys = [
                "status", "daemon", "system", "data",
                "api_budget", "recommendations", "checked_at",
            ]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"
