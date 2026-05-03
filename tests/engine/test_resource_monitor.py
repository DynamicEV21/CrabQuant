"""Tests for resource-aware parallelism (ResourceMonitor)."""

import os
from unittest.mock import patch, MagicMock

import pytest

from crabquant.engine.resource_monitor import (
    DEFAULT_MEMORY_PER_WORKER_MB,
    DEFAULT_RAM_RESERVE_MB,
    DEFAULT_MAX_RAM_USAGE_PCT,
    DEFAULT_MAX_CPU_PCT,
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

    @patch("crabquant.engine.resource_monitor.os.cpu_count", return_value=1)
    @patch("crabquant.engine.resource_monitor.os.statvfs", side_effect=OSError("no statvfs"))
    def test_handles_os_errors_gracefully(self, mock_statvfs, mock_cpu):
        """Snapshot should still be created even if disk/proc reads fail."""
        snap = get_resource_snapshot()
        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_count == 1
        assert snap.disk_free_gb == 0.0


# ── ResourceSnapshot extended tests ──────────────────────────────────────


class TestResourceSnapshotExtended:

    def test_ram_headroom_exact_reserve(self):
        """When available equals reserve, headroom should be 0."""
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=7500.0,
            ram_available_mb=DEFAULT_RAM_RESERVE_MB,
            ram_usage_pct=0.9375,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.ram_headroom_mb == 0.0

    def test_ram_headroom_below_reserve_clamps_to_zero(self):
        """Headroom should never be negative."""
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=7800.0,
            ram_available_mb=100.0,
            ram_usage_pct=0.975,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.ram_headroom_mb == 0.0

    def test_ram_headroom_large_available(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=32000.0,
            ram_used_mb=2000.0,
            ram_available_mb=30000.0,
            ram_usage_pct=0.0625,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.ram_headroom_mb == 30000.0 - DEFAULT_RAM_RESERVE_MB

    @pytest.mark.parametrize("cpu_pct,cpu_count,expected_min", [
        (0.0, 1, 1),
        (0.0, 8, 8),
        (50.0, 4, 2),
        (90.0, 4, 1),
        (100.0, 4, 1),
        (200.0, 4, 1),  # >100% CPU (possible on multi-core psutil)
    ])
    def test_max_workers_by_cpu_parametrized(self, cpu_pct, cpu_count, expected_min):
        snap = ResourceSnapshot(
            cpu_percent=cpu_pct,
            ram_total_mb=16000.0,
            ram_used_mb=4000.0,
            ram_available_mb=12000.0,
            ram_usage_pct=0.25,
            disk_free_gb=100.0,
            cpu_count=cpu_count,
        )
        assert snap.max_workers_by_cpu >= expected_min

    def test_max_workers_by_cpu_single_core_high_load(self):
        snap = ResourceSnapshot(
            cpu_percent=95.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=1,
        )
        # headroom = max(0.1, 1.0 - 0.95) = 0.1; 1 * 0.1 = 0.1 → int = 0 → max(1, 0) = 1
        assert snap.max_workers_by_cpu == 1

    @pytest.mark.parametrize("ram_avail,expected_geq", [
        (500.0, 1),   # 500 - 500 = 0 → max(1, 0) = 1
        (1000.0, 3),  # (1000-500)/150 = 3.33 → 3
        (5000.0, 30), # (5000-500)/150 = 30
    ])
    def test_max_workers_by_ram_parametrized(self, ram_avail, expected_geq):
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=16000.0,
            ram_used_mb=16000.0 - ram_avail,
            ram_available_mb=ram_avail,
            ram_usage_pct=(16000.0 - ram_avail) / 16000.0,
            disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.max_workers_by_ram >= expected_geq

    def test_is_ram_constrained_by_usage_pct(self):
        """Above DEFAULT_MAX_RAM_USAGE_PCT → constrained."""
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=10000.0,
            ram_used_mb=8100.0,
            ram_available_mb=1900.0,
            ram_usage_pct=0.81,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.is_ram_constrained is True

    def test_is_ram_constrained_by_low_available(self):
        """Available < 2 * reserve → constrained even if usage_pct is ok."""
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=100000.0,
            ram_used_mb=99300.0,
            ram_available_mb=700.0,
            ram_usage_pct=0.993,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        # 700 < 2 * 500 = 1000 → constrained
        assert snap.is_ram_constrained is True

    def test_is_ram_constrained_boundary_not_constrained(self):
        """Exactly at threshold should not be constrained."""
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=10000.0,
            ram_used_mb=7990.0,
            ram_available_mb=2010.0,
            ram_usage_pct=0.799,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.is_ram_constrained is False

    def test_is_cpu_constrained_boundary(self):
        """Exactly at 90% CPU → not constrained (strictly greater)."""
        snap = ResourceSnapshot(
            cpu_percent=90.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        assert snap.is_cpu_constrained is False

    def test_to_dict_values_are_rounded(self):
        snap = ResourceSnapshot(
            cpu_percent=55.555,
            ram_total_mb=8123.456,
            ram_used_mb=4000.789,
            ram_available_mb=4122.667,
            ram_usage_pct=0.4925,
            disk_free_gb=123.456,
            cpu_count=4,
        )
        d = snap.to_dict()
        # cpu_percent rounded to 1 decimal
        assert d["cpu_percent"] == 55.6
        # ram values rounded to 0 decimals
        assert d["ram_total_mb"] == 8123.0
        # ram_usage_pct multiplied by 100, rounded to 1 decimal
        assert d["ram_usage_pct"] == 49.2 or d["ram_usage_pct"] == 49.3
        # disk_free_gb rounded to 2 decimals
        assert d["disk_free_gb"] == 123.46

    def test_to_dict_boolean_fields(self):
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=16000.0,
            ram_used_mb=4000.0,
            ram_available_mb=12000.0,
            ram_usage_pct=0.25,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        d = snap.to_dict()
        assert isinstance(d["is_ram_constrained"], bool)
        assert isinstance(d["is_cpu_constrained"], bool)

    def test_to_dict_constrained_scenario(self):
        snap = ResourceSnapshot(
            cpu_percent=95.0,
            ram_total_mb=8000.0,
            ram_used_mb=7200.0,
            ram_available_mb=800.0,
            ram_usage_pct=0.9,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        d = snap.to_dict()
        assert d["is_ram_constrained"] is True
        assert d["is_cpu_constrained"] is True

    def test_default_values(self):
        """Test defaults for optional fields."""
        snap = ResourceSnapshot(
            cpu_percent=0.0,
            ram_total_mb=0.0,
            ram_used_mb=0.0,
            ram_available_mb=0.0,
            ram_usage_pct=0.0,
            disk_free_gb=0.0,
        )
        assert snap.load_avg_1m == 0.0
        assert snap.load_avg_5m == 0.0
        assert snap.cpu_count == 1
        assert snap.timestamp == 0.0

    def test_custom_optional_fields(self):
        snap = ResourceSnapshot(
            cpu_percent=0.0,
            ram_total_mb=0.0,
            ram_used_mb=0.0,
            ram_available_mb=0.0,
            ram_usage_pct=0.0,
            disk_free_gb=0.0,
            load_avg_1m=2.5,
            load_avg_5m=1.8,
            cpu_count=16,
            timestamp=99999.0,
        )
        assert snap.load_avg_1m == 2.5
        assert snap.load_avg_5m == 1.8
        assert snap.cpu_count == 16
        assert snap.timestamp == 99999.0


# ── Constants ────────────────────────────────────────────────────────────


class TestConstants:
    def test_memory_per_worker_positive(self):
        assert DEFAULT_MEMORY_PER_WORKER_MB > 0

    def test_ram_reserve_positive(self):
        assert DEFAULT_RAM_RESERVE_MB > 0

    def test_max_ram_usage_between_zero_and_one(self):
        assert 0.0 < DEFAULT_MAX_RAM_USAGE_PCT <= 1.0

    def test_max_cpu_between_zero_and_one(self):
        assert 0.0 < DEFAULT_MAX_CPU_PCT <= 1.0

    def test_reserve_less_than_per_worker_times_two(self):
        """Reserve should be reasonable relative to per-worker memory."""
        assert DEFAULT_RAM_RESERVE_MB < DEFAULT_MEMORY_PER_WORKER_MB * 10


# ── compute_optimal_workers extended tests ───────────────────────────────


class TestComputeOptimalWorkersExtended:

    def _make_snapshot(self, cpu_pct=50.0, ram_avail=6000.0, ram_total=16000.0, cpu_count=4):
        return ResourceSnapshot(
            cpu_percent=cpu_pct,
            ram_total_mb=ram_total,
            ram_used_mb=ram_total - ram_avail,
            ram_available_mb=ram_avail,
            ram_usage_pct=(ram_total - ram_avail) / ram_total if ram_total > 0 else 0.0,
            disk_free_gb=100.0,
            cpu_count=cpu_count,
        )

    def test_requested_one_always_one(self):
        snap = self._make_snapshot(cpu_pct=99.0, ram_avail=10.0, cpu_count=1)
        result = compute_optimal_workers(1, snap)
        assert result == 1

    def test_zero_total_ram(self):
        """Division by zero protection when ram_total is 0."""
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=0.0,
            ram_used_mb=0.0,
            ram_available_mb=0.0,
            ram_usage_pct=0.0,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        result = compute_optimal_workers(8, snap)
        assert result >= 1

    def test_zero_memory_per_worker(self):
        """Division by zero protection when memory_per_worker_mb is 0."""
        snap = self._make_snapshot(ram_avail=6000.0, cpu_count=4)
        result = compute_optimal_workers(8, snap, memory_per_worker_mb=0.0)
        assert result >= 1

    def test_negative_ram_available(self):
        """Should handle edge case of negative available RAM gracefully."""
        snap = self._make_snapshot(ram_avail=-100.0, cpu_count=4)
        result = compute_optimal_workers(8, snap)
        assert result >= 1

    def test_very_large_requested(self):
        """Very large requested should be clamped."""
        snap = self._make_snapshot(cpu_pct=10.0, ram_avail=10000.0, cpu_count=4)
        result = compute_optimal_workers(10000, snap)
        assert result <= 4  # Limited by cpu_count

    def test_ram_budget_limit_active(self):
        """When current usage already exceeds max_ram_pct budget."""
        snap = self._make_snapshot(
            ram_avail=1000.0, ram_total=10000.0, cpu_count=8,
        )
        # 9000 used / 10000 = 90% usage, max 80% → budget = 8000 - 9000 = negative
        result = compute_optimal_workers(8, snap, max_ram_usage_pct=0.80)
        assert result == 1

    def test_abundant_resources_no_throttle(self):
        snap = self._make_snapshot(cpu_pct=0.0, ram_avail=15000.0, ram_total=16000.0, cpu_count=8)
        result = compute_optimal_workers(4, snap)
        assert result == 4

    @pytest.mark.parametrize("requested", [1, 2, 4, 8, 16])
    def test_various_requested_values(self, requested):
        snap = self._make_snapshot(cpu_pct=20.0, ram_avail=10000.0, cpu_count=8)
        result = compute_optimal_workers(requested, snap)
        assert 1 <= result <= requested

    def test_cpu_at_100_percent(self):
        snap = self._make_snapshot(cpu_pct=100.0, cpu_count=8)
        result = compute_optimal_workers(8, snap)
        # headroom = max(0.1, 1.0 - 1.0) = 0.1; 8 * 0.1 = 0.8 → int = 0 → max(1,0) = 1
        assert result == 1

    def test_single_cpu_system(self):
        snap = self._make_snapshot(cpu_pct=30.0, ram_avail=4000.0, cpu_count=1)
        result = compute_optimal_workers(8, snap)
        assert result == 1


# ── ResourceMonitor extended tests ───────────────────────────────────────


class TestResourceMonitorExtended:

    def test_init_defaults(self):
        monitor = ResourceMonitor()
        assert monitor.check_interval > 0
        assert monitor.memory_per_worker_mb > 0
        assert monitor.min_workers == 1
        assert monitor._last_check == 0.0
        assert monitor._cached_snapshot is None
        assert monitor._last_recommendation == 0
        assert monitor._check_count == 0
        assert monitor._throttle_count == 0

    def test_init_custom_params(self):
        monitor = ResourceMonitor(
            check_interval=10.0,
            memory_per_worker_mb=300.0,
            min_workers=3,
        )
        assert monitor.check_interval == 10.0
        assert monitor.memory_per_worker_mb == 300.0
        assert monitor.min_workers == 3

    def test_check_increments_count(self):
        monitor = ResourceMonitor()
        monitor.check()
        assert monitor._check_count == 1
        monitor.check()
        assert monitor._check_count == 2

    def test_check_updates_timestamp(self):
        monitor = ResourceMonitor()
        import time
        before = time.time()
        monitor.check()
        after = time.time()
        assert before <= monitor._last_check <= after

    def test_check_returns_snapshot(self):
        monitor = ResourceMonitor()
        snap = monitor.check()
        assert isinstance(snap, ResourceSnapshot)

    def test_get_snapshot_refreshes_when_stale(self):
        import time
        monitor = ResourceMonitor(check_interval=0.0)
        snap1 = monitor.check()
        time.sleep(0.01)
        snap2 = monitor.get_snapshot()
        # With check_interval=0, it should always refresh
        assert monitor._check_count >= 2

    def test_get_snapshot_uses_cache_when_fresh(self):
        monitor = ResourceMonitor(check_interval=9999.0)
        snap1 = monitor.check()
        snap2 = monitor.get_snapshot()
        assert snap1 is snap2
        assert monitor._check_count == 1

    def test_get_workers_sets_recommendation(self):
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=16000.0,
            ram_used_mb=4000.0,
            ram_available_mb=12000.0,
            ram_usage_pct=0.25,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        monitor = ResourceMonitor(check_interval=9999.0)
        monitor._cached_snapshot = snap
        workers = monitor.get_workers(4)
        assert monitor._last_recommendation == workers

    def test_get_workers_no_throttle_when_equal(self):
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=16000.0,
            ram_used_mb=4000.0,
            ram_available_mb=12000.0,
            ram_usage_pct=0.25,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        monitor = ResourceMonitor(check_interval=9999.0, min_workers=1)
        monitor._cached_snapshot = snap
        workers = monitor.get_workers(1)
        assert workers == 1
        assert monitor._throttle_count == 0

    def test_context_manager_exit_no_error(self):
        monitor = ResourceMonitor()
        try:
            with monitor:
                pass
        except Exception:
            pytest.fail("Context manager exit raised unexpectedly")

    def test_get_status_structure(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_total_mb=8000.0,
            ram_used_mb=4000.0,
            ram_available_mb=4000.0,
            ram_usage_pct=0.5,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        monitor = ResourceMonitor(check_interval=9999.0)
        monitor._cached_snapshot = snap
        status = monitor.get_status()
        assert status["checks"] == 1  # get_snapshot triggers a check since cache was set but get_status calls get_snapshot
        assert status["throttles"] == 0
        assert status["last_recommendation"] == 0
        assert "cpu_percent" in status["snapshot"]

    def test_multiple_get_workers_calls(self):
        snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_total_mb=16000.0,
            ram_used_mb=4000.0,
            ram_available_mb=12000.0,
            ram_usage_pct=0.25,
            disk_free_gb=100.0,
            cpu_count=4,
        )
        monitor = ResourceMonitor(check_interval=9999.0)
        monitor._cached_snapshot = snap
        w1 = monitor.get_workers(4)
        w2 = monitor.get_workers(4)
        w3 = monitor.get_workers(2)
        assert w1 == w2 == 4
        assert w3 <= 2
