"""Tests for crabquant.refinement.per_wave_metrics — track convergence rate per wave, identify best archetypes."""

import json
from pathlib import Path

import pytest

from crabquant.refinement.per_wave_metrics import (
    WaveMetrics,
    PerWaveMetricsTracker,
    MandateArchetypeStats,
    compute_convergence_rate,
    identify_best_archetypes,
)


class TestWaveMetrics:
    """Test the WaveMetrics dataclass."""

    def test_defaults(self):
        m = WaveMetrics(wave_number=1, total_mandates=5)
        assert m.wave_number == 1
        assert m.total_mandates == 5
        assert m.successful == 0
        assert m.convergence_rate == 0.0
        assert m.avg_sharpe == 0.0

    def test_convergence_rate_calculation(self):
        m = WaveMetrics(wave_number=1, total_mandates=10, successful=7)
        assert m.convergence_rate == pytest.approx(0.7)

    def test_convergence_rate_zero_mandates(self):
        m = WaveMetrics(wave_number=1, total_mandates=0, successful=0)
        assert m.convergence_rate == 0.0

    def test_avg_sharpe_calculation(self):
        m = WaveMetrics(
            wave_number=1,
            total_mandates=3,
            successful=3,
            sharpe_values=[1.0, 2.0, 3.0],
        )
        assert m.avg_sharpe == pytest.approx(2.0)

    def test_avg_sharpe_empty(self):
        m = WaveMetrics(wave_number=1, total_mandates=0)
        assert m.avg_sharpe == 0.0

    def test_to_dict(self):
        m = WaveMetrics(wave_number=2, total_mandates=5, successful=3)
        d = m.to_dict()
        assert d["wave_number"] == 2
        assert d["convergence_rate"] == pytest.approx(0.6)

    def test_from_dict_roundtrip(self):
        original = WaveMetrics(
            wave_number=3,
            total_mandates=10,
            successful=7,
            sharpe_values=[1.5, 2.0, 1.8],
        )
        restored = WaveMetrics.from_dict(original.to_dict())
        assert restored.wave_number == original.wave_number
        assert restored.convergence_rate == original.convergence_rate
        assert restored.sharpe_values == original.sharpe_values

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_convergence_rate_one_hundred_percent(self):
        m = WaveMetrics(wave_number=1, total_mandates=5, successful=5)
        assert m.convergence_rate == pytest.approx(1.0)

    def test_convergence_rate_zero_percent(self):
        m = WaveMetrics(wave_number=1, total_mandates=10, successful=0)
        assert m.convergence_rate == pytest.approx(0.0)

    def test_convergence_rate_one_success_of_many(self):
        m = WaveMetrics(wave_number=1, total_mandates=100, successful=1)
        assert m.convergence_rate == pytest.approx(0.01)

    def test_avg_sharpe_negative_values(self):
        m = WaveMetrics(
            wave_number=1, total_mandates=2, successful=2,
            sharpe_values=[-0.5, -1.5],
        )
        assert m.avg_sharpe == pytest.approx(-1.0)

    def test_avg_sharpe_single_value(self):
        m = WaveMetrics(
            wave_number=1, total_mandates=1, successful=1,
            sharpe_values=[3.14],
        )
        assert m.avg_sharpe == pytest.approx(3.14)

    def test_avg_sharpe_mixed_positive_negative(self):
        m = WaveMetrics(
            wave_number=1, total_mandates=2, successful=2,
            sharpe_values=[2.0, -1.0],
        )
        assert m.avg_sharpe == pytest.approx(0.5)

    def test_to_dict_contains_all_keys(self):
        m = WaveMetrics(wave_number=5, total_mandates=3, successful=2, sharpe_values=[1.0])
        d = m.to_dict()
        expected_keys = {"wave_number", "total_mandates", "successful", "convergence_rate", "avg_sharpe", "sharpe_values"}
        assert set(d.keys()) == expected_keys

    def test_from_dict_defaults_when_missing_optional(self):
        """from_dict uses defaults for missing optional keys."""
        d = {"wave_number": 1, "total_mandates": 5}
        m = WaveMetrics.from_dict(d)
        assert m.successful == 0
        assert m.sharpe_values == []
        assert m.avg_sharpe == 0.0

    def test_from_dict_preserves_sharpe_values(self):
        d = {
            "wave_number": 2,
            "total_mandates": 5,
            "successful": 3,
            "sharpe_values": [0.5, 1.5, 2.5],
        }
        m = WaveMetrics.from_dict(d)
        assert m.sharpe_values == [0.5, 1.5, 2.5]
        assert m.avg_sharpe == pytest.approx(1.5)

    def test_sharpe_values_not_mutated_after_init(self):
        """Changing sharpe_values list after creation doesn't affect avg_sharpe."""
        m = WaveMetrics(wave_number=1, total_mandates=2, sharpe_values=[1.0, 2.0])
        assert m.avg_sharpe == pytest.approx(1.5)
        m.sharpe_values.append(3.0)
        # avg_sharpe was computed in __post_init__, so adding to the list doesn't change it
        assert m.avg_sharpe == pytest.approx(1.5)

    def test_total_mandates_zero_successful_zero(self):
        m = WaveMetrics(wave_number=1, total_mandates=0, successful=0)
        assert m.convergence_rate == 0.0

    def test_wave_number_zero(self):
        m = WaveMetrics(wave_number=0, total_mandates=5, successful=2)
        assert m.wave_number == 0
        assert m.convergence_rate == pytest.approx(0.4)

    def test_large_wave_number(self):
        m = WaveMetrics(wave_number=9999, total_mandates=1, successful=1)
        assert m.wave_number == 9999


class TestComputeConvergenceRate:
    """Test standalone convergence rate computation."""

    def test_basic(self):
        assert compute_convergence_rate(7, 10) == pytest.approx(0.7)

    def test_zero_denominator(self):
        assert compute_convergence_rate(0, 0) == 0.0

    def test_all_successful(self):
        assert compute_convergence_rate(5, 5) == pytest.approx(1.0)

    def test_none_successful(self):
        assert compute_convergence_rate(0, 5) == pytest.approx(0.0)

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_successful_exceeds_total(self):
        """If successful > total (data error), still computes the ratio."""
        rate = compute_convergence_rate(10, 5)
        assert rate == pytest.approx(2.0)

    def test_large_values(self):
        rate = compute_convergence_rate(999, 1000)
        assert rate == pytest.approx(0.999)

    def test_very_small_denominator(self):
        rate = compute_convergence_rate(1, 1)
        assert rate == pytest.approx(1.0)

    def test_fractional_result(self):
        rate = compute_convergence_rate(1, 3)
        assert rate == pytest.approx(1.0 / 3.0)

    def test_returns_float(self):
        rate = compute_convergence_rate(1, 2)
        assert isinstance(rate, float)


class TestPerWaveMetricsTracker:
    """Test the per-wave metrics tracker."""

    def _make_wave_report(self, wave_num=1, successful=3, failed=2, sharpes=None):
        """Create a mock wave report dict."""
        results = []
        if sharpes:
            for i, s in enumerate(sharpes):
                results.append({
                    "mandate_name": f"mandate_{i}",
                    "status": "success" if i < successful else "failed",
                    "sharpe": s,
                    "turns": 5,
                    "archetype": "momentum" if i % 2 == 0 else "mean_reversion",
                })
        else:
            for i in range(successful):
                results.append({
                    "mandate_name": f"mandate_{i}",
                    "status": "success",
                    "sharpe": 1.5 + i * 0.1,
                    "turns": 5,
                    "archetype": "momentum",
                })
            for i in range(failed):
                results.append({
                    "mandate_name": f"mandate_fail_{i}",
                    "status": "failed",
                    "sharpe": 0.0,
                    "turns": 7,
                    "archetype": "momentum",
                    "error": "did not converge",
                })
        return {
            "wave_number": wave_num,
            "total_mandates": successful + failed,
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    def test_record_wave(self):
        tracker = PerWaveMetricsTracker()
        report = self._make_wave_report(wave_num=1, successful=3, failed=2)
        tracker.record_wave(report)
        assert len(tracker.wave_metrics) == 1
        assert tracker.wave_metrics[1].successful == 3
        assert tracker.wave_metrics[1].convergence_rate == pytest.approx(0.6)

    def test_record_multiple_waves(self):
        tracker = PerWaveMetricsTracker()
        tracker.record_wave(self._make_wave_report(wave_num=1, successful=3, failed=2))
        tracker.record_wave(self._make_wave_report(wave_num=2, successful=4, failed=1))
        assert len(tracker.wave_metrics) == 2
        assert tracker.wave_metrics[2].convergence_rate == pytest.approx(0.8)

    def test_sharpe_values_tracked(self):
        tracker = PerWaveMetricsTracker()
        report = self._make_wave_report(
            wave_num=1, successful=3, failed=0,
            sharpes=[1.0, 2.0, 3.0],
        )
        tracker.record_wave(report)
        assert tracker.wave_metrics[1].sharpe_values == [1.0, 2.0, 3.0]
        assert tracker.wave_metrics[1].avg_sharpe == pytest.approx(2.0)

    def test_archetype_tracking(self):
        tracker = PerWaveMetricsTracker()
        report = self._make_wave_report(
            wave_num=1, successful=3, failed=0,
            sharpes=[2.0, 1.5, 1.0],
        )
        tracker.record_wave(report)
        assert "momentum" in tracker.archetype_stats
        assert "mean_reversion" in tracker.archetype_stats

    def test_get_summary(self):
        tracker = PerWaveMetricsTracker()
        tracker.record_wave(self._make_wave_report(wave_num=1, successful=3, failed=2))
        summary = tracker.get_summary()
        assert summary["total_waves"] == 1
        assert "wave_metrics" in summary
        assert "archetype_stats" in summary

    def test_save_and_load(self, tmp_path):
        tracker = PerWaveMetricsTracker()
        tracker.record_wave(self._make_wave_report(wave_num=1, successful=3, failed=2))

        metrics_file = str(tmp_path / "per_wave_metrics.json")
        tracker.save(metrics_file)

        loaded = PerWaveMetricsTracker.load(metrics_file)
        assert len(loaded.wave_metrics) == 1
        assert loaded.wave_metrics[1].successful == 3

    def test_save_creates_directories(self, tmp_path):
        tracker = PerWaveMetricsTracker()
        deep_path = str(tmp_path / "deep" / "metrics.json")
        tracker.save(deep_path)
        assert Path(deep_path).exists()

    def test_load_nonexistent_returns_empty(self, tmp_path):
        tracker = PerWaveMetricsTracker.load(str(tmp_path / "nonexistent.json"))
        assert len(tracker.wave_metrics) == 0
        assert len(tracker.archetype_stats) == 0

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_record_wave_overwrites_same_wave(self):
        """Recording a wave with same number overwrites the previous."""
        tracker = PerWaveMetricsTracker()
        tracker.record_wave(self._make_wave_report(wave_num=1, successful=3, failed=2))
        tracker.record_wave(self._make_wave_report(wave_num=1, successful=5, failed=0))
        assert tracker.wave_metrics[1].successful == 5
        assert tracker.wave_metrics[1].convergence_rate == pytest.approx(1.0)
        assert len(tracker.wave_metrics) == 1

    def test_record_wave_no_results_key(self):
        """Wave report without 'results' key doesn't crash."""
        tracker = PerWaveMetricsTracker()
        report = {"wave_number": 1, "total_mandates": 5, "successful": 3, "failed": 2}
        tracker.record_wave(report)
        assert tracker.wave_metrics[1].sharpe_values == []

    def test_record_wave_empty_results(self):
        """Wave report with empty results list."""
        tracker = PerWaveMetricsTracker()
        report = {"wave_number": 1, "total_mandates": 0, "successful": 0, "results": []}
        tracker.record_wave(report)
        assert tracker.wave_metrics[1].convergence_rate == 0.0

    def test_record_wave_result_without_archetype(self):
        """Results without archetype key don't add to archetype_stats."""
        tracker = PerWaveMetricsTracker()
        report = {
            "wave_number": 1, "total_mandates": 1, "successful": 1,
            "results": [{"status": "success", "sharpe": 1.5}],
        }
        tracker.record_wave(report)
        assert len(tracker.archetype_stats) == 0

    def test_archetype_tracking_across_waves(self):
        """Archetype stats accumulate across multiple waves."""
        tracker = PerWaveMetricsTracker()
        report1 = {
            "wave_number": 1, "total_mandates": 2, "successful": 2,
            "results": [
                {"status": "success", "sharpe": 1.0, "archetype": "momentum"},
                {"status": "success", "sharpe": 2.0, "archetype": "momentum"},
            ],
        }
        report2 = {
            "wave_number": 2, "total_mandates": 1, "successful": 1,
            "results": [
                {"status": "success", "sharpe": 3.0, "archetype": "momentum"},
            ],
        }
        tracker.record_wave(report1)
        tracker.record_wave(report2)
        stats = tracker.archetype_stats["momentum"]
        assert stats.total == 3
        assert stats.successful == 3

    def test_archetype_failed_results_counted(self):
        """Failed archetype results increment total but not successful."""
        tracker = PerWaveMetricsTracker()
        report = {
            "wave_number": 1, "total_mandates": 3, "successful": 1,
            "results": [
                {"status": "success", "sharpe": 1.5, "archetype": "mr"},
                {"status": "failed", "sharpe": 0.0, "archetype": "mr"},
                {"status": "failed", "sharpe": -0.5, "archetype": "mr"},
            ],
        }
        tracker.record_wave(report)
        stats = tracker.archetype_stats["mr"]
        assert stats.total == 3
        assert stats.successful == 1

    def test_get_summary_empty_tracker(self):
        tracker = PerWaveMetricsTracker()
        summary = tracker.get_summary()
        assert summary["total_waves"] == 0
        assert summary["wave_metrics"] == {}
        assert summary["archetype_stats"] == {}

    def test_get_summary_multiple_waves(self):
        tracker = PerWaveMetricsTracker()
        tracker.record_wave(self._make_wave_report(wave_num=1, successful=2, failed=1))
        tracker.record_wave(self._make_wave_report(wave_num=3, successful=4, failed=0))
        summary = tracker.get_summary()
        assert summary["total_waves"] == 2
        assert 1 in summary["wave_metrics"]
        assert 3 in summary["wave_metrics"]

    def test_save_load_roundtrip_with_archetypes(self, tmp_path):
        tracker = PerWaveMetricsTracker()
        report = {
            "wave_number": 1, "total_mandates": 2, "successful": 2,
            "results": [
                {"status": "success", "sharpe": 1.5, "archetype": "momentum"},
                {"status": "success", "sharpe": 2.5, "archetype": "breakout"},
            ],
        }
        tracker.record_wave(report)
        path = str(tmp_path / "metrics.json")
        tracker.save(path)
        loaded = PerWaveMetricsTracker.load(path)
        assert "momentum" in loaded.archetype_stats
        assert "breakout" in loaded.archetype_stats

    def test_save_produces_valid_json(self, tmp_path):
        tracker = PerWaveMetricsTracker()
        tracker.record_wave(self._make_wave_report(wave_num=1, successful=2, failed=1))
        path = str(tmp_path / "metrics.json")
        tracker.save(path)
        data = json.loads(Path(path).read_text())
        assert "total_waves" in data
        assert "wave_metrics" in data

    def test_failed_result_with_missing_sharpe_defaults(self):
        """Failed results without sharpe key use 0.0 default."""
        tracker = PerWaveMetricsTracker()
        report = {
            "wave_number": 1, "total_mandates": 1, "successful": 0,
            "results": [{"status": "failed", "archetype": "test"}],
        }
        tracker.record_wave(report)
        assert tracker.wave_metrics[1].sharpe_values == []
        assert tracker.archetype_stats["test"].total == 1

    def test_result_with_non_success_status(self):
        """Results with status != 'success' don't add sharpe_values."""
        tracker = PerWaveMetricsTracker()
        report = {
            "wave_number": 1, "total_mandates": 2, "successful": 0,
            "results": [
                {"status": "timeout", "sharpe": 0.5, "archetype": "test"},
                {"status": "error", "sharpe": 0.3, "archetype": "test"},
            ],
        }
        tracker.record_wave(report)
        assert tracker.wave_metrics[1].sharpe_values == []
        assert tracker.archetype_stats["test"].successful == 0


class TestMandateArchetypeStats:
    """Test archetype-level aggregation."""

    def test_defaults(self):
        stats = MandateArchetypeStats(archetype="momentum")
        assert stats.total == 0
        assert stats.successful == 0
        assert stats.avg_sharpe == 0.0

    def test_record_success(self):
        stats = MandateArchetypeStats(archetype="momentum")
        stats.record(1.8, success=True)
        assert stats.total == 1
        assert stats.successful == 1
        assert stats.avg_sharpe == pytest.approx(1.8)

    def test_record_failure(self):
        stats = MandateArchetypeStats(archetype="momentum")
        stats.record(0.0, success=False)
        assert stats.total == 1
        assert stats.successful == 0

    def test_to_dict(self):
        stats = MandateArchetypeStats(archetype="momentum")
        stats.record(1.5, success=True)
        d = stats.to_dict()
        assert d["archetype"] == "momentum"
        assert d["successful"] == 1
        assert d["convergence_rate"] == pytest.approx(1.0)

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_record_multiple(self):
        stats = MandateArchetypeStats(archetype="momentum")
        stats.record(1.0, success=True)
        stats.record(2.0, success=True)
        stats.record(0.0, success=False)
        assert stats.total == 3
        assert stats.successful == 2
        assert stats.avg_sharpe == pytest.approx(1.5)

    def test_avg_sharpe_only_successes(self):
        """avg_sharpe should only include successful sharpe values."""
        stats = MandateArchetypeStats(archetype="test")
        stats.record(2.0, success=True)
        stats.record(3.0, success=True)
        stats.record(0.0, success=False)
        # Only 2.0 and 3.0 are in _sharpe_values (successes)
        assert stats.avg_sharpe == pytest.approx(2.5)

    def test_record_failure_does_not_affect_avg_sharpe(self):
        stats = MandateArchetypeStats(archetype="test")
        stats.record(2.0, success=True)
        stats.record(0.0, success=False)
        stats.record(0.0, success=False)
        assert stats.avg_sharpe == pytest.approx(2.0)

    def test_to_dict_contains_convergence_rate(self):
        stats = MandateArchetypeStats(archetype="test")
        stats.record(1.0, success=True)
        stats.record(0.0, success=False)
        d = stats.to_dict()
        assert d["convergence_rate"] == pytest.approx(0.5)

    def test_to_dict_empty_archetype(self):
        stats = MandateArchetypeStats(archetype="empty")
        d = stats.to_dict()
        assert d["total"] == 0
        assert d["convergence_rate"] == 0.0

    def test_repr_does_not_expose_sharpe_values(self):
        """_sharpe_values has repr=False."""
        stats = MandateArchetypeStats(archetype="test")
        stats.record(1.0, success=True)
        r = repr(stats)
        assert "_sharpe_values" not in r

    def test_negative_sharpe_recorded(self):
        stats = MandateArchetypeStats(archetype="test")
        stats.record(-1.5, success=True)
        assert stats.avg_sharpe == pytest.approx(-1.5)

    def test_all_failures_zero_avg_sharpe(self):
        stats = MandateArchetypeStats(archetype="test")
        stats.record(0.0, success=False)
        stats.record(0.0, success=False)
        assert stats.avg_sharpe == 0.0
        assert stats.total == 2


class TestIdentifyBestArchetypes:
    """Test identification of best-performing archetypes."""

    def test_empty_stats(self):
        result = identify_best_archetypes({})
        assert result == []

    def test_single_archetype(self):
        stats = {"momentum": MandateArchetypeStats(archetype="momentum")}
        stats["momentum"].record(1.5, success=True)
        result = identify_best_archetypes(stats)
        assert len(result) == 1
        assert result[0]["archetype"] == "momentum"

    def test_best_first(self):
        stats = {
            "momentum": MandateArchetypeStats(archetype="momentum"),
            "mean_reversion": MandateArchetypeStats(archetype="mean_reversion"),
        }
        stats["momentum"].record(2.0, success=True)
        stats["momentum"].record(1.5, success=True)
        stats["mean_reversion"].record(0.5, success=True)
        stats["mean_reversion"].record(0.0, success=False)

        result = identify_best_archetypes(stats)
        assert result[0]["archetype"] == "momentum"
        assert result[0]["convergence_rate"] > result[1]["convergence_rate"]

    def test_top_n_limit(self):
        stats = {}
        for i in range(5):
            name = f"archetype_{i}"
            stats[name] = MandateArchetypeStats(archetype=name)
            stats[name].record(1.0, success=True)

        result = identify_best_archetypes(stats, top_n=3)
        assert len(result) == 3

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_default_top_n_is_five(self):
        stats = {}
        for i in range(10):
            name = f"archetype_{i}"
            stats[name] = MandateArchetypeStats(archetype=name)
            stats[name].record(1.0, success=True)
        result = identify_best_archetypes(stats)
        assert len(result) == 5

    def test_sort_by_convergence_then_sharpe(self):
        """When convergence rates are equal, sort by avg_sharpe desc."""
        stats = {
            "low_sharpe": MandateArchetypeStats(archetype="low_sharpe"),
            "high_sharpe": MandateArchetypeStats(archetype="high_sharpe"),
        }
        # Both have 100% convergence
        stats["low_sharpe"].record(1.0, success=True)
        stats["high_sharpe"].record(5.0, success=True)

        result = identify_best_archetypes(stats)
        assert result[0]["archetype"] == "high_sharpe"
        assert result[0]["avg_sharpe"] > result[1]["avg_sharpe"]

    def test_top_n_greater_than_available(self):
        stats = {"a": MandateArchetypeStats(archetype="a")}
        stats["a"].record(1.0, success=True)
        result = identify_best_archetypes(stats, top_n=10)
        assert len(result) == 1

    def test_top_n_one(self):
        stats = {
            "a": MandateArchetypeStats(archetype="a"),
            "b": MandateArchetypeStats(archetype="b"),
        }
        stats["a"].record(1.0, success=True)
        stats["b"].record(2.0, success=True)
        result = identify_best_archetypes(stats, top_n=1)
        assert len(result) == 1
        assert result[0]["archetype"] == "b"

    def test_result_dicts_contain_expected_keys(self):
        stats = {"test": MandateArchetypeStats(archetype="test")}
        stats["test"].record(1.5, success=True)
        result = identify_best_archetypes(stats)
        expected_keys = {"archetype", "total", "successful", "convergence_rate", "avg_sharpe"}
        assert set(result[0].keys()) == expected_keys

    def test_all_archetypes_fail(self):
        """Archetypes with 0% convergence still appear in results."""
        stats = {
            "bad_a": MandateArchetypeStats(archetype="bad_a"),
            "bad_b": MandateArchetypeStats(archetype="bad_b"),
        }
        stats["bad_a"].record(0.0, success=False)
        stats["bad_b"].record(0.0, success=False)
        stats["bad_b"].record(0.0, success=False)
        result = identify_best_archetypes(stats)
        # Both have 0% convergence; bad_b has more attempts but same rate
        assert len(result) == 2
        for entry in result:
            assert entry["convergence_rate"] == 0.0
