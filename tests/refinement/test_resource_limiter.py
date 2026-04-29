"""Tests for crabquant.refinement.resource_limiter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_limiter(**overrides):
    """Import and instantiate ResourceLimiter with optional overrides."""
    from crabquant.refinement.resource_limiter import ResourceLimiter

    return ResourceLimiter(**overrides)


def _snapshot(cpu=0.3, ram_gb=8.0, disk_gb=50.0):
    """Build a ResourceSnapshot (bypasses psutil)."""
    from crabquant.refinement.resource_limiter import ResourceSnapshot

    return ResourceSnapshot(
        cpu_percent=cpu,
        ram_available_gb=ram_gb,
        disk_free_gb=disk_gb,
    )


def _set_snapshot(limiter, snap):
    """Inject a snapshot so check_resources is not called again."""
    limiter._last_snapshot = snap


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        rl = _make_limiter()
        assert rl.min_ram_gb == 2.0
        assert rl.min_disk_gb == 1.0
        assert rl.max_parallel == 3
        assert rl.cpu_threshold == 0.85

    def test_custom_values(self):
        rl = _make_limiter(min_ram_gb=4.0, min_disk_gb=2.0, max_parallel=6, cpu_threshold=0.9)
        assert rl.min_ram_gb == 4.0
        assert rl.max_parallel == 6

    def test_invalid_min_ram(self):
        with pytest.raises(ValueError):
            _make_limiter(min_ram_gb=0)

    def test_invalid_min_disk(self):
        with pytest.raises(ValueError):
            _make_limiter(min_disk_gb=-1)

    def test_invalid_max_parallel(self):
        with pytest.raises(ValueError):
            _make_limiter(max_parallel=0)

    def test_invalid_cpu_threshold(self):
        with pytest.raises(ValueError):
            _make_limiter(cpu_threshold=0.0)
        with pytest.raises(ValueError):
            _make_limiter(cpu_threshold=1.5)


# ---------------------------------------------------------------------------
# 2. check_resources() structure
# ---------------------------------------------------------------------------

class TestCheckResources:
    @patch("crabquant.refinement.resource_limiter.psutil")
    def test_returns_expected_keys(self, mock_psutil):
        vm = MagicMock()
        vm.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = vm
        mock_psutil.disk_usage.return_value = MagicMock(free=50 * 1024**3)
        mock_psutil.cpu_percent.return_value = 30.0  # percent, not fraction

        rl = _make_limiter()
        result = rl.check_resources()

        assert "cpu_percent" in result
        assert "ram_free_gb" in result
        assert "disk_free_gb" in result
        assert "status" in result

    @patch("crabquant.refinement.resource_limiter.psutil")
    def test_values_rounded(self, mock_psutil):
        vm = MagicMock()
        vm.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = vm
        mock_psutil.disk_usage.return_value = MagicMock(free=50 * 1024**3)
        mock_psutil.cpu_percent.return_value = 30.5

        rl = _make_limiter()
        result = rl.check_resources()

        assert result["cpu_percent"] == 0.305
        assert result["ram_free_gb"] == 8.0
        assert result["disk_free_gb"] == 50.0

    @patch("crabquant.refinement.resource_limiter.psutil")
    def test_status_ok_when_resources_abundant(self, mock_psutil):
        vm = MagicMock()
        vm.available = 16 * 1024**3
        mock_psutil.virtual_memory.return_value = vm
        mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024**3)
        mock_psutil.cpu_percent.return_value = 20.0

        rl = _make_limiter()
        result = rl.check_resources()
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# 3. should_pause()
# ---------------------------------------------------------------------------

class TestShouldPause:
    def test_pause_when_ram_below_min(self):
        rl = _make_limiter(min_ram_gb=2.0)
        _set_snapshot(rl, _snapshot(ram_gb=1.5))
        assert rl.should_pause() is True

    def test_no_pause_when_ram_ok(self):
        rl = _make_limiter(min_ram_gb=2.0)
        _set_snapshot(rl, _snapshot(ram_gb=5.0, disk_gb=10.0))
        assert rl.should_pause() is False

    def test_pause_when_disk_below_min(self):
        rl = _make_limiter(min_disk_gb=1.0)
        _set_snapshot(rl, _snapshot(ram_gb=8.0, disk_gb=0.5))
        assert rl.should_pause() is True

    def test_no_pause_when_disk_ok(self):
        rl = _make_limiter(min_disk_gb=1.0)
        _set_snapshot(rl, _snapshot(ram_gb=8.0, disk_gb=5.0))
        assert rl.should_pause() is False


# ---------------------------------------------------------------------------
# 4. get_recommended_parallel()
# ---------------------------------------------------------------------------

class TestRecommendedParallel:
    def test_returns_max_when_abundant(self):
        rl = _make_limiter(max_parallel=4)
        _set_snapshot(rl, _snapshot(cpu=0.3, ram_gb=8.0, disk_gb=50.0))
        assert rl.get_recommended_parallel() == 4

    def test_zero_when_ram_critical(self):
        rl = _make_limiter(max_parallel=4, min_ram_gb=2.0)
        _set_snapshot(rl, _snapshot(ram_gb=1.0))
        assert rl.get_recommended_parallel() == 0

    def test_zero_when_disk_critical(self):
        rl = _make_limiter(max_parallel=4, min_disk_gb=1.0)
        _set_snapshot(rl, _snapshot(ram_gb=8.0, disk_gb=0.5))
        assert rl.get_recommended_parallel() == 0

    def test_cpu_constrained_reduces_by_one(self):
        rl = _make_limiter(max_parallel=4, cpu_threshold=0.85)
        _set_snapshot(rl, _snapshot(cpu=0.9, ram_gb=8.0, disk_gb=50.0))
        assert rl.get_recommended_parallel() == 3

    def test_cpu_constrained_minimum_one(self):
        rl = _make_limiter(max_parallel=1, cpu_threshold=0.85)
        _set_snapshot(rl, _snapshot(cpu=0.9, ram_gb=8.0, disk_gb=50.0))
        assert rl.get_recommended_parallel() == 1

    def test_ram_constrained_reduces_by_one(self):
        rl = _make_limiter(max_parallel=4)
        _set_snapshot(rl, _snapshot(cpu=0.3, ram_gb=3.0, disk_gb=50.0))
        assert rl.get_recommended_parallel() == 3

    def test_both_cpu_and_ram_constrained(self):
        rl = _make_limiter(max_parallel=4, cpu_threshold=0.85)
        _set_snapshot(rl, _snapshot(cpu=0.9, ram_gb=3.0, disk_gb=50.0))
        # CPU reduces by 1 → 3, RAM reduces by 1 → 2
        assert rl.get_recommended_parallel() == 2


# ---------------------------------------------------------------------------
# 5. /proc/meminfo fallback
# ---------------------------------------------------------------------------

class TestProcMeminfoFallback:
    def test_fallback_returns_bytes(self):
        from crabquant.refinement.resource_limiter import _ram_from_proc_meminfo

        sample = "MemAvailable:    2048000 kB\nMemTotal:    8192000 kB\n"
        with patch(
            "crabquant.refinement.resource_limiter.Path.read_text",
            return_value=sample,
        ):
            result = _ram_from_proc_meminfo()
        assert result == 2048000 * 1024  # kB → bytes

    def test_fallback_returns_none_on_missing_key(self):
        from crabquant.refinement.resource_limiter import _ram_from_proc_meminfo

        sample = "MemTotal:    8192000 kB\n"
        with patch(
            "crabquant.refinement.resource_limiter.Path.read_text",
            return_value=sample,
        ):
            result = _ram_from_proc_meminfo()
        assert result is None

    def test_fallback_returns_none_on_os_error(self):
        from crabquant.refinement.resource_limiter import _ram_from_proc_meminfo

        with patch(
            "crabquant.refinement.resource_limiter.Path.read_text",
            side_effect=OSError("no /proc"),
        ):
            result = _ram_from_proc_meminfo()
        assert result is None


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_ram_triggers_pause(self):
        rl = _make_limiter()
        _set_snapshot(rl, _snapshot(ram_gb=0.0))
        assert rl.should_pause() is True
        assert rl.get_recommended_parallel() == 0

    def test_zero_disk_triggers_pause(self):
        rl = _make_limiter()
        _set_snapshot(rl, _snapshot(ram_gb=8.0, disk_gb=0.0))
        assert rl.should_pause() is True

    def test_exact_boundary_ram(self):
        rl = _make_limiter(min_ram_gb=2.0)
        # Exactly at boundary → should NOT pause
        _set_snapshot(rl, _snapshot(ram_gb=2.0, disk_gb=10.0))
        assert rl.should_pause() is False

    def test_just_below_boundary_ram(self):
        rl = _make_limiter(min_ram_gb=2.0)
        _set_snapshot(rl, _snapshot(ram_gb=1.999, disk_gb=10.0))
        assert rl.should_pause() is True


# ---------------------------------------------------------------------------
# 7. get_status_summary()
# ---------------------------------------------------------------------------

class TestStatusSummary:
    def test_summary_structure(self):
        rl = _make_limiter()
        _set_snapshot(rl, _snapshot())
        summary = rl.get_status_summary()

        expected_keys = {
            "cpu_percent",
            "ram_free_gb",
            "disk_free_gb",
            "recommended_parallel",
            "should_pause",
            "max_parallel",
            "min_ram_gb",
            "min_disk_gb",
            "cpu_threshold",
        }
        assert set(summary.keys()) == expected_keys

    def test_summary_values_match(self):
        rl = _make_limiter(max_parallel=4)
        _set_snapshot(rl, _snapshot(cpu=0.9, ram_gb=3.0, disk_gb=50.0))
        summary = rl.get_status_summary()

        assert summary["max_parallel"] == 4
        assert summary["should_pause"] is False
        # CPU constrained (-1) + RAM constrained (-1) = 2
        assert summary["recommended_parallel"] == 2
