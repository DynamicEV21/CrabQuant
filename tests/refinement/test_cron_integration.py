"""Tests for scripts/crabquant_cron.py — Phase 3 cron integration."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ── We need to import from scripts/ which may not be on sys.path ──────────────

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")


def _import_crabquant_cron():
    """Import crabquant_cron from scripts/ dir."""
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    import crabquant_cron
    return crabquant_cron


crabquant_cron = _import_crabquant_cron()


parse_mandate_configs = crabquant_cron.parse_mandate_configs
run_single_mandate = crabquant_cron.run_single_mandate
run_wave = crabquant_cron.run_wave
find_pending_mandates = crabquant_cron.find_pending_mandates
collect_wave_results = crabquant_cron.collect_wave_results
CRON_RUNS_DIR = crabquant_cron.CRON_RUNS_DIR
MANDATES_DIR = crabquant_cron.MANDATES_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mandate(name: str = "test_mandate", **overrides) -> dict:
    mandate = {
        "name": name,
        "description": "Test mandate",
        "strategy_archetype": "momentum",
        "tickers": ["SPY"],
        "primary_ticker": "SPY",
        "period": "2y",
        "sharpe_target": 1.5,
        "max_turns": 3,
        "constraints": {},
    }
    mandate.update(overrides)
    return mandate


# ── parse_mandate_configs ─────────────────────────────────────────────────────

class TestParseMandateConfigs:

    def test_parses_single_mandate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mandate_path = Path(tmpdir) / "m1.json"
            mandate_path.write_text(json.dumps(make_mandate("m1")))

            mandates = parse_mandate_configs(tmpdir)
            assert len(mandates) == 1
            assert mandates[0]["name"] == "m1"

    def test_parses_multiple_mandates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["m1", "m2", "m3"]:
                Path(tmpdir, f"{name}.json").write_text(json.dumps(make_mandate(name)))

            mandates = parse_mandate_configs(tmpdir)
            assert len(mandates) == 3

    def test_skips_non_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "m1.json").write_text(json.dumps(make_mandate("m1")))
            Path(tmpdir, "readme.txt").write_text("not a mandate")
            Path(tmpdir, "state.yaml").write_text("key: value")

            mandates = parse_mandate_configs(tmpdir)
            assert len(mandates) == 1

    def test_returns_empty_for_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mandates = parse_mandate_configs(tmpdir)
            assert mandates == []

    def test_returns_empty_for_nonexistent_dir(self):
        mandates = parse_mandate_configs("/nonexistent/dir")
        assert mandates == []

    def test_validates_mandate_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "bad.json").write_text('{"name": "x"}')
            mandates = parse_mandate_configs(tmpdir)
            # Should still include it (validation happens later)
            assert len(mandates) == 1


# ── find_pending_mandates ─────────────────────────────────────────────────────

class TestFindPendingMandates:

    def test_finds_mandates_not_yet_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mandate
            m_dir = Path(tmpdir) / "mandates"
            m_dir.mkdir()
            m_dir.joinpath("m1.json").write_text(json.dumps(make_mandate("m1")))

            # Create runs dir with NO state for m1
            r_dir = Path(tmpdir) / "runs"
            r_dir.mkdir()

            pending = find_pending_mandates(str(m_dir), str(r_dir))
            assert len(pending) == 1
            assert pending[0]["name"] == "m1"

    def test_skips_already_completed_mandates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m_dir = Path(tmpdir) / "mandates"
            m_dir.mkdir()
            m_dir.joinpath("m1.json").write_text(json.dumps(make_mandate("m1")))

            r_dir = Path(tmpdir) / "runs"
            run_dir = r_dir / "2026-04-26_100000_m1"
            run_dir.mkdir(parents=True)
            state = {"run_id": "test", "mandate_name": "m1", "status": "success",
                     "created_at": "2026-04-26T10:00:00"}
            run_dir.joinpath("state.json").write_text(json.dumps(state))

            pending = find_pending_mandates(str(m_dir), str(r_dir))
            assert len(pending) == 0

    def test_finds_stale_running_mandates(self):
        """Mandates with status='running' but old timestamp should be considered pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            m_dir = Path(tmpdir) / "mandates"
            m_dir.mkdir()
            m_dir.joinpath("m1.json").write_text(json.dumps(make_mandate("m1")))

            r_dir = Path(tmpdir) / "runs"
            run_dir = r_dir / "2026-04-25_100000_m1"  # old date
            run_dir.mkdir(parents=True)
            state = {"run_id": "test", "mandate_name": "m1", "status": "running",
                     "created_at": "2026-04-25T10:00:00"}
            run_dir.joinpath("state.json").write_text(json.dumps(state))

            pending = find_pending_mandates(str(m_dir), str(r_dir))
            assert len(pending) == 1


# ── run_single_mandate ────────────────────────────────────────────────────────

class TestRunSingleMandate:

    @patch("crabquant_cron.subprocess.run")
    def test_calls_refinement_loop(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Run directory: /tmp/run_test\nSuccess!",
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_single_mandate(
                "/tmp/mandate.json",
                runs_dir=tmpdir,
                timeout=60,
            )

        assert result["status"] in ("success", "error", "unknown")
        assert "mandate" in result

    @patch("crabquant_cron.subprocess.run")
    def test_handles_timeout(self, mock_subprocess):
        mock_subprocess.side_effect = subprocess.TimeoutExpired(
            cmd="test", timeout=60
        )

        result = run_single_mandate("/tmp/mandate.json", timeout=60)
        assert result["status"] == "error"
        assert "timeout" in result.get("error", "").lower()

    @patch("crabquant_cron.subprocess.run")
    def test_handles_nonzero_exit(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Something went wrong",
        )

        result = run_single_mandate("/tmp/mandate.json", timeout=60)
        assert result["status"] == "error"


# ── run_wave ──────────────────────────────────────────────────────────────────

class TestRunWave:

    @patch("crabquant_cron.run_single_mandate")
    def test_runs_all_mandates(self, mock_run):
        mock_run.return_value = {
            "mandate": "test",
            "status": "success",
            "sharpe": 1.5,
            "turns": 3,
        }

        mandates = [make_mandate(f"m{i}") for i in range(3)]
        report = run_wave(mandates, max_parallel=2, timeout=60)

        assert report["total"] == 3
        assert report["successful"] == 3
        assert report["failed"] == 0
        assert mock_run.call_count == 3

    @patch("crabquant_cron.run_single_mandate")
    def test_counts_failures(self, mock_run):
        mock_run.return_value = {
            "mandate": "test",
            "status": "error",
            "sharpe": 0,
            "turns": 0,
            "error": "crashed",
        }

        mandates = [make_mandate("m1")]
        report = run_wave(mandates, max_parallel=1, timeout=60)

        assert report["total"] == 1
        assert report["failed"] == 1
        assert report["successful"] == 0

    @patch("crabquant_cron.run_single_mandate")
    def test_empty_mandate_list(self, mock_run):
        report = run_wave([], max_parallel=1, timeout=60)
        assert report["total"] == 0
        mock_run.assert_not_called()


# ── collect_wave_results ─────────────────────────────────────────────────────

class TestCollectWaveResults:

    def test_collects_from_runs_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two completed runs
            for name, status, sharpe in [("m1", "success", 1.8), ("m2", "max_turns", 0.9)]:
                run_dir = Path(tmpdir) / f"run_{name}"
                run_dir.mkdir()
                state = {
                    "run_id": f"run_{name}",
                    "mandate_name": name,
                    "status": status,
                    "best_sharpe": sharpe,
                    "current_turn": 3,
                    "created_at": "2026-04-26T10:00:00",
                    "history": [],
                }
                run_dir.joinpath("state.json").write_text(json.dumps(state))

            results = collect_wave_results(tmpdir)

            assert len(results) == 2
            assert results[0]["mandate"] == "m1"
            assert results[1]["mandate"] == "m2"

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = collect_wave_results(tmpdir)
            assert results == []

    def test_skips_dirs_without_state_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "empty_dir").mkdir()
            results = collect_wave_results(tmpdir)
            assert results == []
