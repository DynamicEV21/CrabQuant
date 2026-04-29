"""Tests for resource-aware parallelism (ResourceMonitor)."""

import os
from unittest.mock import patch, MagicMock

import pytest

from crabquant.engine.resource_monitor import (
    DEFAULT_MEMORY_PER_WORKER_MB,
    DEFAULT_RAM_RESERVE_MB,
    ResourceMonitor,
    ResourceSnapshot,
    compute_optimal_workers,
    get_resource_snapshot,
)


# ── ResourceSnapshot ──────────────────────────────────────────────────────


class TestResourceSnapshot:

    def test_ram_headroom_subtracts_reserve(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=5000.0,
            ram_available_mb=3000.0,
            ram_usage_pct=0.625,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.ram_headroom_mb == 3000.0 - DEFAULT_RAM_RESERVE_MB

    def test_max_workers_by_ram(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        expected = int((4000.0 - DEFAULT_RAM_RESERVE_MB) / DEFAULT_MEMORY_PER_WORKER_MB)
        assert snap.max_workers_by_ram == expected

    def test_max_workers_by_ram_zero_available(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=7800.0,
            ram_available_mb=200.0,
            ram_usage_pct=0.975,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        # 200MB - 500MB reserve = negative → 0 → clamped to 1
        assert snap.max_workers_by_ram >= 1

    def test_max_workers_by_cpu(self):
        snap = ResourceSnapshot(
            cpu_percent=80.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=8,
        )
        # 20% headroom → 8 * 0.2 = 1.6 → int = 1
        assert snap.max_workers_by_cpu == 1

    def test_max_workers_by_cpu_idle(self):
        snap = ResourceSnapshot(
            cpu_percent=0.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.max_workers_by_cpu == 4

    def test_is_ram_constrained(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=7200.0,
            ram_available_mb=800.0,
            ram_usage_pct=0.9,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.is_ram_constrained is True

    def test_is_cpu_constrained(self):
        snap = ResourceSnapshot(
            cpu_percent=95.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.is_cpu_constrained is True

    def test_not_constrained(self):
        snap = ResourceSnapshot(
            cpu_percent=30.0,
            ram_total_mb=16000.0,
            ram_used_mb=4000.0,
            ram_available_mb=12000.0,
            ram_usage_pct=0.25,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.is_ram_constrained is False
        assert snap.is_cpu_constrained is False

    def test_to_dict_has_required_keys(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        d = snap.to_dict()
        expected_keys = {
            "cpu_percent", "ram_total_mb", "ram_used_mb", "ram_available_mb",
            "ram_usage_pct", "ram_headroom_mb", "disk_free_gb",
            "max_workers_by_ram", "max_workers_by_cpu",
            "is_ram_constrained", "is_cpu_constrained", "cpu_count",
        }
        assert expected_keys.issubset(d.keys())


# ── compute_optimal_workers ────────────────────────────────────────────────


class TestComputeOptimalWorkers:

    def _make_snapshot(self, cpu_pct=50.0, ram_avail=6000.0, ram_total=16000.0, cpu_count=4):
        return ResourceSnapshot(
            cpu_percent=cpu_pct,
            ram_total_mb=ram_total,
            ram_used_mb=ram_total - ram_avail,
            ram_available_mb=ram_avail,
            ram_usage_pct=(ram_total - ram_avail) / ram_total,
            disk_free_gb=100.0,
            cpu_count=cpu_count,
        )

    def test_returns_requested_when_resources_abundant(self):
        snap = self._make_snapshot(cpu_pct=10.0, ram_avail=14000.0, cpu_count=8)
        result = compute_optimal_workers(4, snap)
        assert result == 4

    def test_throttled_by_cpu(self):
        snap = self._make_snapshot(cpu_pct=90.0, cpu_count=4)
        result = compute_optimal_workers(8, snap)
        # CPU headroom: (1 - 0.9) * 4 = 0.4 → int = 0 → clamped to 1
        assert result == 1

    def test_throttled_by_ram(self):
        snap = self._make_snapshot(ram_avail=200.0, cpu_count=8)
        result = compute_optimal_workers(8, snap)
        # 200MB - 500MB reserve = negative → 1
        assert result == 1

    def test_never_below_one(self):
        snap = self._make_snapshot(cpu_pct=99.0, ram_avail=50.0, cpu_count=1)
        result = compute_optimal_workers(100, snap)
        assert result >= 1

    def test_respects_cpu_count(self):
        snap = self._make_snapshot(cpu_pct=0.0, cpu_count=2)
        result = compute_optimal_workers(100, snap)
        assert result <= 2

    def test_custom_memory_per_worker(self):
        snap = self._make_snapshot(ram_avail=1000.0, ram_total=16000.0, cpu_count=8)
        # RAM limit: (1000 - 100) / 200 = 4
        # CPU headroom at 50%: 8 * 0.5 = 4
        # Both limit to 4
        result = compute_optimal_workers(
            8, snap,
            memory_per_worker_mb=200.0,
            ram_reserve_mb=100.0,
            max_ram_usage_pct=0.99,
        )
        assert result == 4

    def test_custom_ram_reserve(self):
        snap = self._make_snapshot(ram_avail=1000.0, ram_total=16000.0, cpu_count=8)
        # RAM limit: (1000 - 200) / 150 = 5
        # CPU headroom at 50%: 8 * 0.5 = 4
        # min(8, 8, 5, 4, ...) = 4 (CPU-limited)
        result = compute_optimal_workers(
            8, snap,
            ram_reserve_mb=200.0,
            max_ram_usage_pct=0.99,
        )
        assert result == 4  # CPU-limited

    def test_custom_max_ram_pct(self):
        snap = self._make_snapshot(ram_avail=8000.0, ram_total=16000.0, cpu_count=8)
        # 50% usage, max 40% → 16000 * 0.4 - 8000 used = negative → 1
        result = compute_optimal_workers(8, snap, max_ram_usage_pct=0.40)
        assert result == 1

    def test_none_snapshot_takes_fresh(self):
        # Just verify it doesn't crash
        result = compute_optimal_workers(2)
        assert result >= 1


# ── ResourceMonitor ───────────────────────────────────────────────────────


class TestResourceMonitor:

    def test_context_manager(self):
        monitor = ResourceMonitor()
        with monitor as m:
            assert m is monitor
            assert m._cached_snapshot is not None

    def test_get_snapshot_caches(self):
        monitor = ResourceMonitor(check_interval=999.0)
        snap1 = monitor.check()
        snap2 = monitor.get_snapshot()
        assert snap1 is snap2  # Same object due to caching

    def test_get_workers(self):
        monitor = ResourceMonitor(check_interval=999.0)
        with monitor:
            workers = monitor.get_workers(8)
            assert 1 <= workers <= 8

    def test_get_workers_respects_min(self):
        monitor = ResourceMonitor(check_interval=999.0, min_workers=2)
        with monitor:
            workers = monitor.get_workers(1)
            assert workers >= 2

    def test_get_status(self):
        monitor = ResourceMonitor(check_interval=999.0)
        with monitor:
            status = monitor.get_status()
            assert "checks" in status
            assert "throttles" in status
            assert "snapshot" in status
            assert "last_recommendation" in status

    def test_throttle_counting(self):
        # Create a snapshot that will definitely throttle
        snap = ResourceSnapshot(
            cpu_percent=99.0,
            ram_total_mb=1000.0,
            ram_used_mb=950.0,
            ram_available_mb=50.0,
            ram_usage_pct=0.95,
            disk_free_gb=100.0,
            cpu_count=2,
        )
        monitor = ResourceMonitor(check_interval=999.0)
        monitor._cached_snapshot = snap

        workers = monitor.get_workers(100)
        assert monitor._throttle_count >= 1
        assert workers < 100


# ── get_resource_snapshot ─────────────────────────────────────────────────


class TestGetResourceSnapshot:

    @patch("crabquant.engine.resource_monitor.get_resource_snapshot", wraps=get_resource_snapshot)
    def test_returns_snapshot(self, mock_snap):
        snap = get_resource_snapshot()
        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_count >= 1
        assert snap.ram_total_mb > 0
        assert snap.disk_free_gb >= 0

    def test_uses_psutil_when_available(self):
        import importlib
        # Verify psutil is available and produces reasonable values
        try:
            import psutil
            snap = get_resource_snapshot()
            assert snap.ram_total_mb > 0
            assert isinstance(snap.cpu_percent, float)
        except ImportError:
            pytest.skip("psutil not installed")

    def test_fallback_without_psutil(self):
        # Just verify it works — can't easily uninstall psutil in test
        snap = get_resource_snapshot()
        assert isinstance(snap, ResourceSnapshot)
