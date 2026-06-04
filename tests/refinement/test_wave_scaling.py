"""Tests for crabquant.refinement.wave_scaling — increased parallel limit and wave status tracking."""

import json
from unittest.mock import MagicMock, patch

import pytest

from crabquant.refinement.wave_scaling import (
    WaveStatus,
    WaveStatusTracker,
    SCALING_CONFIG,
    get_parallel_limit,
    get_wave_status_summary,
)


class TestScalingConfig:
    """Test the default scaling configuration."""

    def test_default_parallel_limit_is_5(self):
        assert SCALING_CONFIG["default_parallel_limit"] == 5

    def test_max_parallel_limit(self):
        assert SCALING_CONFIG["max_parallel_limit"] == 10

    def test_has_required_keys(self):
        for key in ("default_parallel_limit", "max_parallel_limit", "status_file"):
            assert key in SCALING_CONFIG


class TestGetParallelLimit:
    """Test parallel limit calculation."""

    def test_default_limit(self):
        assert get_parallel_limit() == 5

    def test_explicit_override(self):
        assert get_parallel_limit(override=8) == 8

    def test_clamped_to_max(self):
        assert get_parallel_limit(override=20) == 10

    def test_minimum_one(self):
        assert get_parallel_limit(override=0) == 1

    def test_negative_override_clamped_to_one(self):
        assert get_parallel_limit(override=-5) == 1

    def test_float_override_converted_to_int(self):
        assert get_parallel_limit(override=7.9) == 7

    def test_exact_max_value(self):
        assert get_parallel_limit(override=10) == 10

    def test_one_above_max_clamped(self):
        assert get_parallel_limit(override=11) == 10

    def test_override_of_one(self):
        assert get_parallel_limit(override=1) == 1


class TestWaveStatus:
    """Test the WaveStatus dataclass."""

    def test_defaults(self):
        status = WaveStatus(wave_number=1)
        assert status.wave_number == 1
        assert status.status == "pending"
        assert status.mandate_count == 0
        assert status.completed_count == 0
        assert status.successful_count == 0
        assert status.failed_count == 0
        assert status.started_at == ""
        assert status.completed_at == ""

    def test_to_dict(self):
        status = WaveStatus(wave_number=2, status="running", mandate_count=5)
        d = status.to_dict()
        assert d["wave_number"] == 2
        assert d["status"] == "running"
        assert d["mandate_count"] == 5

    def test_from_dict_roundtrip(self):
        original = WaveStatus(
            wave_number=3,
            status="completed",
            mandate_count=5,
            completed_count=5,
            successful_count=3,
            failed_count=2,
        )
        restored = WaveStatus.from_dict(original.to_dict())
        assert restored == original

    def test_convergence_rate_zero_mandate_count(self):
        status = WaveStatus(wave_number=1, mandate_count=0, successful_count=0)
        assert status.convergence_rate == 0.0

    def test_convergence_rate_all_successful(self):
        status = WaveStatus(wave_number=1, mandate_count=10, successful_count=10)
        assert status.convergence_rate == 1.0

    def test_convergence_rate_partial(self):
        status = WaveStatus(wave_number=1, mandate_count=10, successful_count=7)
        assert status.convergence_rate == pytest.approx(0.7)

    def test_to_dict_includes_convergence_rate(self):
        status = WaveStatus(wave_number=1, mandate_count=4, successful_count=2)
        d = status.to_dict()
        assert "convergence_rate" in d
        assert d["convergence_rate"] == pytest.approx(0.5)

    def test_from_dict_with_missing_keys_uses_defaults(self):
        d = {"wave_number": 42}
        ws = WaveStatus.from_dict(d)
        assert ws.wave_number == 42
        assert ws.status == "pending"
        assert ws.mandate_count == 0
        assert ws.successful_count == 0
        assert ws.error == ""

    def test_from_dict_preserves_error_field(self):
        d = {"wave_number": 1, "error": "OOM killed"}
        ws = WaveStatus.from_dict(d)
        assert ws.error == "OOM killed"

    def test_from_dict_preserves_timestamps(self):
        d = {
            "wave_number": 1,
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T01:00:00",
        }
        ws = WaveStatus.from_dict(d)
        assert ws.started_at == "2025-01-01T00:00:00"
        assert ws.completed_at == "2025-01-01T01:00:00"


class TestWaveStatusTracker:
    """Test the wave status tracker."""

    def test_start_wave(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=5)
        assert tracker.current_wave == 1
        assert tracker.waves[1].status == "running"
        assert tracker.waves[1].mandate_count == 5

    def test_complete_wave(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=5)
        tracker.complete_wave(wave_number=1, successful=3, failed=2)
        assert tracker.waves[1].status == "completed"
        assert tracker.waves[1].successful_count == 3
        assert tracker.waves[1].failed_count == 2
        assert tracker.waves[1].completed_count == 5

    def test_fail_wave(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=3)
        tracker.fail_wave(wave_number=1, error="API timeout")
        assert tracker.waves[1].status == "failed"
        assert "timeout" in tracker.waves[1].error.lower()

    def test_multiple_waves(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=5)
        tracker.complete_wave(wave_number=1, successful=3, failed=2)
        tracker.start_wave(wave_number=2, mandate_count=5)
        tracker.complete_wave(wave_number=2, successful=4, failed=1)
        assert len(tracker.waves) == 2
        assert tracker.current_wave == 2

    def test_get_status_summary(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=5)
        tracker.complete_wave(wave_number=1, successful=3, failed=2)
        summary = tracker.get_status_summary()
        assert summary["total_waves"] == 1
        assert summary["total_successful"] == 3
        assert summary["total_failed"] == 2
        assert summary["overall_convergence_rate"] == pytest.approx(0.6)

    def test_save_and_load(self, tmp_path):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=5)
        tracker.complete_wave(wave_number=1, successful=3, failed=2)

        status_file = str(tmp_path / "wave_status.json")
        tracker.save(status_file)

        loaded = WaveStatusTracker.load(status_file)
        assert loaded.current_wave == 1
        assert loaded.waves[1].successful_count == 3

    def test_save_creates_directories(self, tmp_path):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=3)
        deep_path = str(tmp_path / "a" / "b" / "wave_status.json")
        tracker.save(deep_path)
        assert (tmp_path / "a" / "b" / "wave_status.json").exists()

    def test_load_nonexistent_returns_empty(self, tmp_path):
        tracker = WaveStatusTracker.load(str(tmp_path / "nonexistent.json"))
        assert tracker.current_wave == 0
        assert len(tracker.waves) == 0

    def test_complete_nonexistent_wave_is_noop(self):
        tracker = WaveStatusTracker()
        tracker.complete_wave(wave_number=99, successful=1, failed=0)
        assert len(tracker.waves) == 0

    def test_fail_nonexistent_wave_is_noop(self):
        tracker = WaveStatusTracker()
        tracker.fail_wave(wave_number=99, error="boom")
        assert len(tracker.waves) == 0

    def test_start_wave_overwrites_previous(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=3)
        tracker.complete_wave(wave_number=1, successful=2, failed=1)
        # Re-start same wave resets it
        tracker.start_wave(wave_number=1, mandate_count=5)
        assert tracker.waves[1].status == "running"
        assert tracker.waves[1].mandate_count == 5
        assert tracker.waves[1].successful_count == 0

    def test_get_status_summary_empty_tracker(self):
        tracker = WaveStatusTracker()
        summary = tracker.get_status_summary()
        assert summary["total_waves"] == 0
        assert summary["total_successful"] == 0
        assert summary["total_failed"] == 0
        assert summary["total_mandates"] == 0
        assert summary["overall_convergence_rate"] == 0.0
        assert summary["waves"] == {}

    def test_get_status_summary_multiple_waves(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=10)
        tracker.complete_wave(wave_number=1, successful=7, failed=3)
        tracker.start_wave(wave_number=2, mandate_count=10)
        tracker.complete_wave(wave_number=2, successful=5, failed=5)
        summary = tracker.get_status_summary()
        assert summary["total_waves"] == 2
        assert summary["total_successful"] == 12
        assert summary["total_failed"] == 8
        assert summary["total_mandates"] == 20
        assert summary["overall_convergence_rate"] == pytest.approx(0.6)

    def test_save_and_load_multiple_waves(self, tmp_path):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=5)
        tracker.complete_wave(wave_number=1, successful=3, failed=2)
        tracker.start_wave(wave_number=3, mandate_count=8)
        tracker.complete_wave(wave_number=3, successful=6, failed=2)

        status_file = str(tmp_path / "multi.json")
        tracker.save(status_file)
        loaded = WaveStatusTracker.load(status_file)
        assert loaded.current_wave == 3
        assert len(loaded.waves) == 2
        assert loaded.waves[3].successful_count == 6

    def test_load_corrupt_json_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("NOT JSON {{{")
        tracker = WaveStatusTracker.load(str(bad_file))
        assert tracker.current_wave == 0
        assert len(tracker.waves) == 0

    def test_load_json_missing_waves_key(self, tmp_path):
        bad_file = tmp_path / "no_waves.json"
        bad_file.write_text(json.dumps({"total_waves": 5}))
        tracker = WaveStatusTracker.load(str(bad_file))
        assert len(tracker.waves) == 0

    def test_save_produces_valid_json(self, tmp_path):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=3)
        tracker.fail_wave(wave_number=1, error="crash")
        path = str(tmp_path / "out.json")
        tracker.save(path)
        data = json.loads((tmp_path / "out.json").read_text())
        assert "total_waves" in data
        assert "1" in data["waves"]

    def test_get_status_summary_waves_are_string_keys(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=7, mandate_count=1)
        summary = tracker.get_status_summary()
        assert "7" in summary["waves"]
        assert summary["waves"]["7"]["wave_number"] == 7


class TestGetWaveStatusSummary:
    """Test the standalone summary function."""

    def test_empty_tracker(self):
        tracker = WaveStatusTracker()
        summary = get_wave_status_summary(tracker)
        assert summary["total_waves"] == 0
        assert summary["total_successful"] == 0
        assert summary["overall_convergence_rate"] == 0.0

    def test_with_results(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=10)
        tracker.complete_wave(wave_number=1, successful=7, failed=3)
        summary = get_wave_status_summary(tracker)
        assert summary["total_waves"] == 1
        assert summary["overall_convergence_rate"] == pytest.approx(0.7)

    def test_is_alias_for_tracker_method(self):
        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=4)
        tracker.complete_wave(wave_number=1, successful=4, failed=0)
        assert get_wave_status_summary(tracker) == tracker.get_status_summary()
