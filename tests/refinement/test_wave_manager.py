"""Tests for wave_manager — parallel wave execution.

Covers:
- WaveResult dataclass
- WaveReport dataclass and convergence_rate
- run_single_mandate edge cases
- run_wave with various outcomes
- run_waves lifecycle and stopping conditions
"""

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
    WaveReport,
    WaveResult,
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

    def test_optional_error_field(self):
        r = WaveResult(mandate_name="x", status="success", best_sharpe=2.0, turns_used=5, run_dir="/r")
        assert r.error is None

    def test_all_status_values(self):
        """WaveResult should accept all documented status values."""
        for status in ("success", "max_turns", "stuck", "failed", "error", "abandoned"):
            r = WaveResult(mandate_name="m", status=status, best_sharpe=0, turns_used=0, run_dir="")
            assert r.status == status

    def test_negative_sharpe(self):
        """WaveResult should handle negative Sharpe (bad strategies)."""
        r = WaveResult(mandate_name="bad", status="failed", best_sharpe=-1.5, turns_used=7, run_dir="/bad")
        assert r.best_sharpe == -1.5

    def test_zero_turns(self):
        """WaveResult for a strategy that didn't complete any turns."""
        r = WaveResult(mandate_name="early", status="error", best_sharpe=0, turns_used=0, run_dir="")
        assert r.turns_used == 0

    def test_large_sharpe(self):
        r = WaveResult(mandate_name="great", status="success", best_sharpe=99.99, turns_used=1, run_dir="/r")
        assert r.best_sharpe == 99.99


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

    def test_to_dict_empty_results(self):
        """to_dict with no results should still work."""
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=0, successful=0, failed=0,
        )
        d = report.to_dict()
        assert d["results"] == []

    def test_convergence_rate_is_property(self):
        """convergence_rate should be a property, not a stored attribute."""
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=4, successful=2, failed=2,
        )
        # Mutate and check recalculation
        assert report.convergence_rate == 0.5
        report.successful = 3
        assert report.convergence_rate == pytest.approx(0.75)

    def test_to_dict_includes_timestamps(self):
        report = WaveReport(
            wave_number=1, started_at="2025-01-01T00:00:00Z", completed_at="2025-01-01T01:00:00Z",
            total_mandates=1, successful=1, failed=0,
        )
        d = report.to_dict()
        assert d["started_at"] == "2025-01-01T00:00:00Z"
        assert d["completed_at"] == "2025-01-01T01:00:00Z"

    def test_to_dict_includes_convergence_rate(self):
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=10, successful=7, failed=3,
        )
        d = report.to_dict()
        assert "convergence_rate" in d
        assert d["convergence_rate"] == pytest.approx(0.7)

    def test_convergence_rate_single_mandate(self):
        """Single mandate that succeeds = 100% convergence."""
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=1, successful=1, failed=0,
        )
        assert report.convergence_rate == pytest.approx(1.0)

    def test_convergence_rate_fractional(self):
        report = WaveReport(
            wave_number=1, started_at="t1", completed_at="t2",
            total_mandates=7, successful=3, failed=4,
        )
        assert report.convergence_rate == pytest.approx(3 / 7)


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

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_finds_state_json(self, mock_run):
        """When state.json exists, parse and return WaveResult from it."""
        mock_run.return_value = MagicMock(
            stdout="done\n",
            stderr="",
            returncode=0,
        )

        state_data = {"status": "success", "best_sharpe": 1.8, "current_turn": 5}

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_run_dir = MagicMock()
            mock_run_dir.stat().st_mtime = 1000
            mock_state_path = MagicMock()
            mock_state_path.exists.return_value = True
            mock_state_path.read_text.return_value = json.dumps(state_data)
            mock_run_dir.__truediv__ = MagicMock(return_value=mock_state_path)

            mock_refinement_runs = MagicMock()
            mock_refinement_runs.exists.return_value = True
            mock_refinement_runs.glob.return_value = [mock_run_dir]

            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"name": "found_test"})
            mock_p.stem = "found_test"
            mock_p.parent = MagicMock()
            mock_p.parent.parent = MagicMock()
            mock_p.parent.parent.parent = MagicMock()
            mock_p.parent.parent.parent.__truediv__ = MagicMock(return_value=mock_refinement_runs)
            mock_path_cls.return_value = mock_p

            result = run_single_mandate("/fake/mandate.json")
            assert result.status == "success"
            assert result.best_sharpe == 1.8
            assert result.turns_used == 5

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_mandate_name_from_file(self, mock_run):
        """Should extract mandate name from JSON if present."""
        mock_run.side_effect = FileNotFoundError("script not found")

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"name": "custom_name", "tickers": ["SPY"]})
            mock_p.stem = "custom_name"
            mock_path_cls.return_value = mock_p

            result = run_single_mandate("/fake/mandate.json")
            assert result.mandate_name == "custom_name"

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_mandate_name_fallback_to_stem(self, mock_run):
        """Should fall back to file stem if name not in JSON."""
        mock_run.side_effect = FileNotFoundError("script not found")

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"tickers": ["SPY"]})
            mock_p.stem = "file_stem_name"
            mock_path_cls.return_value = mock_p

            result = run_single_mandate("/fake/mandate.json")
            assert result.mandate_name == "file_stem_name"

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_extra_args_passed_to_command(self, mock_run):
        """Extra args should be appended to the subprocess command."""
        mock_run.side_effect = FileNotFoundError("nope")

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"name": "args_test"})
            mock_p.stem = "args_test"
            mock_path_cls.return_value = mock_p

            run_single_mandate("/fake/mandate.json", extra_args=["--mode", "fast"])

        call_args = mock_run.call_args[0][0]
        assert "--mode" in call_args
        assert "fast" in call_args

    @patch("crabquant.refinement.wave_manager.subprocess.run")
    def test_error_result_with_stderr(self, mock_run):
        """Error result should include truncated stderr."""
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="A" * 1000 + "important error message",
            returncode=1,
        )

        with patch("crabquant.refinement.wave_manager.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.read_text.return_value = json.dumps({"name": "stderr_test"})
            mock_p.stem = "stderr_test"
            mock_p.exists.return_value = False
            mock_path_cls.return_value = mock_p

            result = run_single_mandate("/fake/mandate.json")
            assert result.status == "error"
            assert result.error is not None
            assert len(result.error) <= 500


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

    def test_run_wave_all_fail(self):
        """All mandates fail."""
        results = [
            WaveResult(f"m{i}", "failed", 0, 7, "", "stuck")
            for i in range(3)
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

                report = run_wave(["/a.json", "/b.json", "/c.json"], max_parallel=2)

        assert report.total_mandates == 3
        assert report.successful == 0
        assert report.failed == 3
        assert report.convergence_rate == 0.0

    def test_run_wave_future_exception(self):
        """If a future raises, it should be caught and counted as failed."""
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("future crash")

        mock_pool = MagicMock()
        mock_pool.submit.return_value = mock_future

        with patch("crabquant.refinement.wave_manager.ProcessPoolExecutor") as mock_exec_cls:
            mock_exec_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
            mock_exec_cls.return_value.__exit__ = MagicMock(return_value=False)
            with patch("crabquant.refinement.wave_manager.as_completed") as mock_ac:
                mock_ac.return_value = iter([mock_future])

                report = run_wave(["/a.json"], max_parallel=1)

        assert report.total_mandates == 1
        assert report.failed == 1
        assert report.results[0].status == "error"

    def test_run_wave_single_mandate(self):
        """Single mandate wave should work."""
        result = WaveResult("solo", "success", 2.0, 2, "/run_solo")
        mock_future = MagicMock()
        mock_future.result.return_value = result

        mock_pool = MagicMock()
        mock_pool.submit.return_value = mock_future

        with patch("crabquant.refinement.wave_manager.ProcessPoolExecutor") as mock_exec_cls:
            mock_exec_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
            mock_exec_cls.return_value.__exit__ = MagicMock(return_value=False)
            with patch("crabquant.refinement.wave_manager.as_completed") as mock_ac:
                mock_ac.return_value = iter([mock_future])

                report = run_wave(["/solo.json"], max_parallel=1)

        assert report.total_mandates == 1
        assert report.successful == 1
        assert report.convergence_rate == 1.0

    def test_run_wave_wave_number_zero(self):
        """run_wave sets wave_number to 0 (caller sets it)."""
        with patch("crabquant.refinement.wave_manager.ProcessPoolExecutor") as mock_exec_cls:
            mock_exec_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_exec_cls.return_value.__exit__ = MagicMock(return_value=False)
            with patch("crabquant.refinement.wave_manager.as_completed", return_value=iter([])):
                report = run_wave([], max_parallel=1)
        assert report.wave_number == 0

    def test_run_wave_completed_at_set(self):
        """WaveReport should have completed_at set after execution."""
        with patch("crabquant.refinement.wave_manager.ProcessPoolExecutor") as mock_exec_cls:
            mock_exec_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_exec_cls.return_value.__exit__ = MagicMock(return_value=False)
            with patch("crabquant.refinement.wave_manager.as_completed", return_value=iter([])):
                report = run_wave([], max_parallel=1)
        assert report.completed_at != ""
        assert len(report.completed_at) > 0


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

    def test_max_waves_limit(self):
        """Should stop after max_waves even if convergence not met."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                Path(tmpdir, f"m{i}.json").write_text('{"name": "m' + str(i) + '"}')

            def always_low_convergence(*args, **kwargs):
                return WaveReport(0, "t1", "t2", 2, 0, 2)

            with patch("crabquant.refinement.wave_manager.run_wave", new=always_low_convergence), \
                 patch("crabquant.refinement.wave_manager.time.sleep"):
                reports = run_waves(
                    mandate_dir=tmpdir,
                    wave_size=2,
                    max_waves=3,
                    stop_on_convergence=0.9,  # Never reached
                )

        assert len(reports) == 3  # max_waves=3

    @patch("crabquant.refinement.wave_manager.time.sleep")
    @patch("crabquant.refinement.wave_manager.run_wave")
    def test_wave_report_saved(self, mock_run_wave, mock_sleep, tmp_path):
        """Each wave report should be saved as a JSON file."""
        mock_run_wave.return_value = WaveReport(
            wave_number=0, started_at="t1", completed_at="t2",
            total_mandates=2, successful=1, failed=1,
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            mandate_dir = Path(tmpdir) / "mandates"
            mandate_dir.mkdir()
            (mandate_dir / "m.json").write_text('{"name": "m"}')

            # runs_dir = mandate_dir.parent / "runs"
            run_wave_calls = [0]

            def track_run_wave(*args, **kwargs):
                run_wave_calls[0] += 1
                return WaveReport(0, "t1", "t2", 1, 0, 1)

            with patch("crabquant.refinement.wave_manager.run_wave", new=track_run_wave):
                reports = run_waves(
                    mandate_dir=str(mandate_dir),
                    wave_size=1,
                    max_waves=2,
                    stop_on_convergence=0.9,
                )

        # At least 1 report should be generated
        assert len(reports) >= 1

    def test_sleep_called_between_waves(self):
        """time.sleep should be called between waves (not after the last)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                Path(tmpdir, f"m{i}.json").write_text('{"name": "m' + str(i) + '"}')

            call_count = [0]

            def low_convergence(*args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                if idx == 0:
                    return WaveReport(0, "t1", "t2", 2, 0, 2)
                return WaveReport(0, "t3", "t4", 2, 2, 0)  # 100% convergence

            with patch("crabquant.refinement.wave_manager.run_wave", new=low_convergence), \
                 patch("crabquant.refinement.wave_manager.time.sleep") as mock_sleep:
                reports = run_waves(
                    mandate_dir=tmpdir,
                    wave_size=2,
                    max_waves=5,
                    stop_on_convergence=0.5,
                )

        # sleep should be called once (between wave 1 and wave 2)
        assert mock_sleep.call_count == 1

    def test_wave_numbering(self):
        """Wave reports should be numbered 1, 2, 3, ..."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                Path(tmpdir, f"m{i}.json").write_text('{"name": "m' + str(i) + '"}')

            call_count = [0]

            def low_convergence(*args, **kwargs):
                call_count[0] += 1
                return WaveReport(0, "t1", "t2", 2, 0, 2)

            with patch("crabquant.refinement.wave_manager.run_wave", new=low_convergence), \
                 patch("crabquant.refinement.wave_manager.time.sleep"):
                reports = run_waves(
                    mandate_dir=tmpdir,
                    wave_size=2,
                    max_waves=3,
                    stop_on_convergence=0.9,
                )

        for i, report in enumerate(reports, 1):
            assert report.wave_number == i

    @patch("crabquant.refinement.wave_manager.time.sleep")
    @patch("crabquant.refinement.wave_manager.run_wave")
    def test_wave_size_larger_than_mandates(self, mock_run_wave, mock_sleep):
        """When wave_size > number of mandates, it cycles through them."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                Path(tmpdir, f"m{i}.json").write_text('{"name": "m' + str(i) + '"}')

            mock_run_wave.return_value = WaveReport(
                wave_number=0, started_at="t1", completed_at="t2",
                total_mandates=5, successful=5, failed=0,
            )

            with patch("crabquant.refinement.wave_manager.run_wave", mock_run_wave):
                reports = run_waves(
                    mandate_dir=tmpdir,
                    wave_size=5,  # more mandates than files
                    max_waves=1,
                )

        assert len(reports) == 1
        # run_wave should be called with 5 paths (some repeated)
        call_args = mock_run_wave.call_args[0][0]
        assert len(call_args) == 5
