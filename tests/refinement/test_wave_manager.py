"""Tests for wave_manager — parallel wave execution."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Add scripts dir to path for refinement_loop import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root))

from crabquant.refinement.wave_manager import (
    WaveResult,
    WaveReport,
    run_single_mandate,
    run_wave,
    run_waves,
)


# ── WaveResult ───────────────────────────────────────────────────────────────

class TestWaveResult:
    def test_creation_defaults(self):
        r = WaveResult(mandate_name="test", status="success", best_sharpe=1.5, turns_used=3, run_dir="/tmp/run1")
        assert r.mandate_name == "test"
        assert r.status == "success"
        assert r.best_sharpe == 1.5
        assert r.turns_used == 3
        assert r.error is None

    def test_creation_with_error(self):
        r = WaveResult(mandate_name="test", status="error", best_sharpe=0, turns_used=0, run_dir="", error="boom")
        assert r.error == "boom"


# ── WaveReport ───────────────────────────────────────────────────────────────

class TestWaveReport:
    def test_convergence_rate_success(self):
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=5, successful=3, failed=2,
        )
        assert report.convergence_rate == pytest.approx(0.6)

    def test_convergence_rate_zero(self):
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=5, successful=0, failed=5,
        )
        assert report.convergence_rate == 0.0

    def test_convergence_rate_all(self):
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=3, successful=3, failed=0,
        )
        assert report.convergence_rate == pytest.approx(1.0)

    def test_convergence_rate_no_mandates(self):
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=0, successful=0, failed=0,
        )
        assert report.convergence_rate == 0.0

    def test_to_dict(self):
        report = WaveReport(
            wave_number=2, started_at="t1", completed_at="t2",
            total_mandates=2, successful=1, failed=1,
            results=[
                WaveResult("a", "success", 1.0, 3, "/run_a"),
                WaveResult("b", "error", 0, 0, "", "fail"),
            ],
        )
        d = report.to_dict()
        assert d["wave_number"] == 2
        assert d["total_mandates"] == 2
        assert d["successful"] == 1
        assert d["convergence_rate"] == 0.5
        assert len(d["results"]) == 2
        assert d["results"][0]["mandate"] == "a"
        assert d["results"][1]["error"] == "fail"


# ── run_single_mandate ──────────────────────────────────────────────────────

class TestRunSingleMandate:
    def _make_mandate_path(self, tmp_path, name="test_mandate"):
        """Create a temp mandate file and return its path."""
        mandate_file = tmp_path / f"{name}.json"
        mandate_file.write_text(json.dumps({
            "name": name,
            "tickers": ["AAPL"],
            "period": "1y",
        }))
        return str(mandate_file)

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_timeout_returns_error(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="test", timeout=600)

        result = run_single_mandate.__wrapped__ if hasattr(run_single_mandate, '__wrapped__') else None
        # Patch Path to return our mandate content
        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"name": "timeout_test"})
            mock_p.stem = "timeout_test"
            mock_path_cls.return_value = mock_p

            result = run_single_mandate("/fake/mandate.json")
            assert result.status == "error"
            assert "timed out" in result.error.lower()

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_exception_returns_error(self, mock_run):
        mock_run.side_effect = RuntimeError("process crash")

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"name": "crash_test"})
            mock_p.stem = "crash_test"
            mock_path_cls.return_value = mock_p

            result = run_single_mandate("/fake/mandate.json")
            assert result.status == "error"
            assert "process crash" in result.error

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_no_run_directory_found(self, mock_run):
        """When subprocess succeeds but no state.json is found, return error."""
        mock_run.return_value = MagicMock(
            stdout="Some output without run directory info\n",
            stderr="",
            returncode=0,
        )

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"name": "no_dir_test"})
            mock_p.stem = "no_dir_test"
            # refinement_runs doesn't exist
            mock_p.exists.return_value = False
            mock_path_cls.return_value = mock_p

            result = run_single_mandate("/fake/mandate.json")
            assert result.status == "error"
            assert "No run directory" in result.error or "error" in result.status.lower()


# ── run_wave ─────────────────────────────────────────────────────────────────

class TestRunWave:
    def test_run_wave_success_and_error(self):
        """run_wave collects results from futures and counts success/failure."""
        result_success = WaveResult("a", "success", 1.5, 3, "/run_a")
        result_error = WaveResult("b", "error", 0, 0, "", "failed")

        mock_future1 = MagicMock()
        mock_future1.result.return_value = result_success
        mock_future2 = MagicMock()
        mock_future2.result.return_value = result_error

        mock_pool = MagicMock()
        mock_pool.submit.side_effect = [mock_future1, mock_future2]

        with patch("crabquant.refinement.wave_manager.ProcessPoolExecutor") as mock_exec_cls:
            mock_exec_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
            mock_exec_cls.return_value.__exit__ = MagicMock(return_value=False)
            with patch("crabquant.refinement.wave_manager.as_completed") as mock_ac:
                mock_ac.return_value = iter([mock_future1, mock_future2])

                report = run_wave(["/a.json", "/b.json"], max_parallel=2)

        assert report.total_mandates == 2
        assert report.successful == 1
        assert report.failed == 1
        assert len(report.results) == 2

    def test_run_wave_all_success(self):
        results = [
            WaveResult(f"m{i}", "success", 1.0 + i * 0.5, 3, f"/run{i}")
            for i in range(4)
        ]

        futures = []
        for r in results:
            f = MagicMock()
            f.result.return_value = r
            futures.append(f)

        mock_pool = MagicMock()
        mock_pool.submit.side_effect = futures

        with patch("crabquant.refinement.wave_manager.ProcessPoolExecutor") as mock_exec_cls:
            mock_exec_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
            mock_exec_cls.return_value.__exit__ = MagicMock(return_value=False)
            with patch("crabquant.refinement.wave_manager.as_completed") as mock_ac:
                mock_ac.return_value = iter(futures)

                report = run_wave(["/a.json", "/b.json", "/c.json", "/d.json"], max_parallel=4)

        assert report.total_mandates == 4
        assert report.successful == 4
        assert report.failed == 0
        assert report.convergence_rate == 1.0


# ── run_waves ────────────────────────────────────────────────────────────────

class TestRunWaves:
    @patch("crabquant.refinement.wave_manager.time.sleep")
    @patch("crabquant.refinement.wave_manager.run_wave")
    def test_single_wave(self, mock_run_wave, mock_sleep):
        mock_run_wave.return_value = WaveReport(
            wave_number=0, started_at="t1", completed_at="t2",
            total_mandates=3, successful=2, failed=1,
        )

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            # sorted(Path(mandate_dir).glob("*.json")) should return a list of Paths
            mock_mandate = MagicMock()
            mock_mandate.__str__.return_value = "/mandates/a.json"
            mock_mandate.__lt__ = lambda self, other: True

            mock_root = MagicMock()
            mock_root.glob.return_value = [mock_mandate]
            mock_root.parent = MagicMock()
            mock_root.parent.parent = MagicMock()
            mock_root.parent.parent.mkdir = MagicMock()

            mock_path_cls.return_value = mock_root

            reports = run_waves(
                mandate_dir="/mandates",
                max_parallel=2,
                wave_size=3,
                max_waves=1,
            )

        assert len(reports) == 1
        assert reports[0].wave_number == 1

    @patch("crabquant.refinement.wave_manager.time.sleep")
    @patch("crabquant.refinement.wave_manager.run_wave")
    def test_stops_on_convergence(self, mock_run_wave, mock_sleep):
        mock_run_wave.return_value = WaveReport(
            wave_number=0, started_at="t1", completed_at="t2",
            total_mandates=5, successful=5, failed=0,
        )

        # Use real temp files so sorted() works on Path objects
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                Path(tmpdir, f"m{i}.json").write_text('{"name": "m' + str(i) + '"}')

            with patch("crabquant.refinement.wave_manager.run_wave", mock_run_wave):
                reports = run_waves(
                    mandate_dir=tmpdir,
                    stop_on_convergence=0.9,
                    max_waves=10,
                    wave_size=5,
                )

        # Should stop after 1 wave since convergence is 100%
        assert len(reports) == 1
        assert mock_run_wave.call_count == 1

    @patch("crabquant.refinement.wave_manager.time.sleep")
    @patch("crabquant.refinement.wave_manager.run_wave")
    def test_empty_mandate_dir(self, mock_run_wave, mock_sleep):
        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_root = MagicMock()
            mock_root.glob.return_value = []
            mock_path_cls.return_value = mock_root

            reports = run_waves(mandate_dir="/empty")
            assert len(reports) == 0
            mock_run_wave.assert_not_called()

    def test_multiple_waves(self):
        """Run multiple waves when convergence isn't met on first wave."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                Path(tmpdir, f"m{i}.json").write_text('{"name": "m' + str(i) + '"}')

            call_count = [0]

            def fake_run_wave(*args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                # Wave 1: 1/3 = 33% (below 50% target)
                # Wave 2: 3/3 = 100% (above 50% target → stop)
                if idx == 0:
                    return WaveReport(0, "t1", "t2", 3, 1, 2)
                return WaveReport(0, "t3", "t4", 3, 3, 0)

            with patch("crabquant.refinement.wave_manager.run_wave", new=fake_run_wave), \
                 patch("crabquant.refinement.wave_manager.time.sleep"):
                reports = run_waves(
                    mandate_dir=tmpdir,
                    wave_size=3,
                    max_waves=5,
                    stop_on_convergence=0.5,
                )

        assert len(reports) == 2
        assert reports[0].successful == 1
        assert reports[1].successful == 3
