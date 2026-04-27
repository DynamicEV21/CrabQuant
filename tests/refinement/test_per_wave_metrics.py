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
