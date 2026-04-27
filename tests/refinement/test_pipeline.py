"""Integration tests for the pipeline daemon (scripts/run_pipeline.py)."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Import the module under test — we need to manipulate PID_FILE paths
# so tests don't collide with a real daemon.
from scripts import run_pipeline as rp  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────


def _use_tmp_pid(tmp_path: Path) -> None:
    """Redirect PID_FILE, STATE_FILE, HEARTBEAT_FILE into tmp_path."""
    rp.PID_FILE = tmp_path / "crabquant.pid"
    rp.STATE_FILE = tmp_path / "daemon_state.json"
    rp.HEARTBEAT_FILE = tmp_path / "daemon_heartbeat.json"


def _make_pid(pid: int, path: Path) -> None:
    """Write a PID file with the given value."""
    path.write_text(str(pid))


# ── PID management ────────────────────────────────────────────────────────


class TestPIDManagement:
    """Tests for acquire_pid / release_pid."""

    def test_pid_management(self, tmp_path: Path) -> None:
        """acquire_pid writes PID file, release_pid deletes it."""
        _use_tmp_pid(tmp_path)
        assert rp.acquire_pid() is True
        assert rp.PID_FILE.exists()
        assert rp.PID_FILE.read_text().strip() == str(os.getpid())

        rp.release_pid()
        assert not rp.PID_FILE.exists()

    def test_acquire_pid_refuses_double(self, tmp_path: Path) -> None:
        """Second acquire_pid returns False when first holds PID (same process)."""
        _use_tmp_pid(tmp_path)
        assert rp.acquire_pid() is True
        # A second call from the *same* process — os.kill(pid, 0) succeeds
        # so acquire_pid returns False.
        assert rp.acquire_pid() is False
        rp.release_pid()

    def test_acquire_pid_stale(self, tmp_path: Path) -> None:
        """Stale PID (dead process) is overwritten."""
        _use_tmp_pid(tmp_path)
        # Use a PID that almost certainly doesn't exist
        _make_pid(999999, rp.PID_FILE)
        assert rp.acquire_pid() is True
        assert rp.PID_FILE.read_text().strip() == str(os.getpid())
        rp.release_pid()


# ── Stop daemon ───────────────────────────────────────────────────────────


class TestStopDaemon:
    def test_stop_daemon_not_running(self, tmp_path: Path, capsys) -> None:
        """stop_daemon with no PID file prints message."""
        _use_tmp_pid(tmp_path)
        rp.stop_daemon()
        assert "No daemon running" in capsys.readouterr().out

    def test_stop_daemon(self, tmp_path: Path, capsys) -> None:
        """stop_daemon sends SIGTERM to a running child and waits."""
        _use_tmp_pid(tmp_path)

        # Start a short-lived child that ignores SIGTERM briefly, then exits
        child = subprocess.Popen(
            [sys.executable, "-c", "import time, signal; signal.signal(signal.SIGTERM, lambda *a: None); time.sleep(30)"],
            cwd=str(project_root),
        )
        _make_pid(child.pid, rp.PID_FILE)

        # stop_daemon sends SIGTERM then polls for 10s — but our child
        # ignores SIGTERM, so we kill it manually after a short wait.
        # We'll run stop_daemon in a thread to avoid blocking the test.
        import threading

        t = threading.Thread(target=rp.stop_daemon)
        t.start()

        # Wait for SIGTERM to be delivered, then force-kill
        time.sleep(1.5)
        child.kill()
        child.wait()
        t.join(timeout=12)

        output = capsys.readouterr().out
        assert "Sent SIGTERM" in output


# ── Status ────────────────────────────────────────────────────────────────


class TestPrintStatus:
    def test_print_status_stopped(self, tmp_path: Path, capsys) -> None:
        """print_status shows STOPPED when no PID file exists."""
        _use_tmp_pid(tmp_path)
        rp.print_status()
        output = capsys.readouterr().out
        assert "STOPPED" in output
        assert "No state file found" in output

    def test_print_status_running(self, tmp_path: Path, capsys) -> None:
        """print_status shows RUNNING with valid PID and state."""
        _use_tmp_pid(tmp_path)
        from crabquant.refinement.state import DaemonState

        state = DaemonState.create()
        state.current_wave = 3
        state.total_mandates_run = 10
        state.total_strategies_promoted = 2
        state.pending_mandates = ["foo.json"]
        state.save(str(rp.STATE_FILE))

        _make_pid(os.getpid(), rp.PID_FILE)
        rp.print_status()
        output = capsys.readouterr().out
        assert "RUNNING" in output
        assert "Wave: 3" in output
        assert "Mandates run: 10" in output


# ── Mandate discovery ─────────────────────────────────────────────────────


class TestDiscoverMandates:
    def test_discover_mandates(self, tmp_path: Path) -> None:
        """Finds .json files in directory."""
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")
        (tmp_path / "c.txt").write_text("")  # should be ignored
        result = rp.discover_mandates(tmp_path)
        assert result == ["a.json", "b.json"]

    def test_discover_mandates_empty(self, tmp_path: Path) -> None:
        """Returns empty list for empty dir."""
        tmp_path.mkdir(exist_ok=True)
        result = rp.discover_mandates(tmp_path)
        assert result == []

    def test_discover_mandates_missing_dir(self, tmp_path: Path) -> None:
        """Returns empty list for non-existent directory."""
        result = rp.discover_mandates(tmp_path / "nope")
        assert result == []


# ── Single mandate runner ─────────────────────────────────────────────────


class TestRunSingleMandate:
    @patch("scripts.run_pipeline.subprocess.run")
    def test_run_single_mandate_success(self, mock_run: MagicMock) -> None:
        """Run a mandate via subprocess, check return dict structure."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="all good", stderr=""
        )
        result = rp.run_single_mandate("mandates/test.json")
        assert result["returncode"] == 0
        assert "stdout" in result
        assert "stderr" in result

    @patch("scripts.run_pipeline.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="", timeout=1))
    def test_run_single_mandate_timeout(self, mock_run: MagicMock) -> None:
        """Timeout returns special returncode."""
        result = rp.run_single_mandate("mandates/test.json", timeout=1)
        assert result["returncode"] == -1
        assert result["stderr"] == "TIMEOUT"


# ── Signal handler ────────────────────────────────────────────────────────


class TestSignalHandler:
    def test_signal_handler_sets_flag(self) -> None:
        """SIGTERM sets shutdown_requested = True."""
        rp.shutdown_requested = False
        rp.handle_signal(signal.SIGTERM, None)
        assert rp.shutdown_requested is True

    def test_sigint_sets_flag(self) -> None:
        """SIGINT also sets shutdown_requested = True."""
        rp.shutdown_requested = False
        rp.handle_signal(signal.SIGINT, None)
        assert rp.shutdown_requested is True


# ── Daemon max-waves ──────────────────────────────────────────────────────


class TestDaemonMaxWaves:
    @patch("scripts.run_pipeline.run_single_mandate")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_max_waves_0(
        self, mock_sleep: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Daemon with max_waves=0 and no pending mandates just sleeps."""
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()

        mock_run.return_value = {"returncode": 0, "stdout": "", "stderr": ""}
        # Make sleep set shutdown_requested after 1 call to avoid infinite loop
        def sleep_then_stop(*args, **kwargs):
            rp.shutdown_requested = True
        mock_sleep.side_effect = sleep_then_stop

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=0, parallel=1, sleep_seconds=1, mandates_dir=mandates_dir)

        # Should have slept at least once (no mandates found)
        assert mock_sleep.called

    @patch("scripts.run_pipeline.run_single_mandate")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_max_waves_1(
        self, mock_sleep: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Daemon with max_waves=1 runs 1 wave and exits."""
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()
        (mandates_dir / "test.json").write_text('{"name": "test"}')

        mock_run.return_value = {"returncode": 0, "stdout": "", "stderr": ""}

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=1, parallel=1, sleep_seconds=1, mandates_dir=mandates_dir)

        # Should have run the single mandate exactly once
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "test.json" in call_args[0][0]

    @patch("scripts.run_pipeline.run_single_mandate")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_graceful_shutdown(
        self, mock_sleep: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Daemon exits cleanly when shutdown_requested is set mid-wave."""
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()
        (mandates_dir / "a.json").write_text('{"name": "a"}')
        (mandates_dir / "b.json").write_text('{"name": "b"}')

        call_count = 0

        def fake_run(mandate_path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # After first mandate, signal shutdown
                rp.shutdown_requested = True
            return {"returncode": 0, "stdout": "", "stderr": ""}

        mock_run.side_effect = fake_run

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=0, parallel=2, sleep_seconds=60, mandates_dir=mandates_dir)

        # Only first mandate should have run
        assert call_count == 1
