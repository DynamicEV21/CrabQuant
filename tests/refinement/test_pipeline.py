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
        assert result["stderr"].startswith("TIMEOUT")


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


def _make_fake_popen(returncode=0, stdout="", stderr="", trigger_shutdown=False):
    """Create a mock Popen that finishes immediately on poll()."""
    mock_proc = MagicMock()
    mock_proc.returncode = returncode
    # Use a plain function for poll to avoid mock recursion
    if trigger_shutdown:
        def poll_shutdown(*args, **kwargs):
            rp.shutdown_requested = True
            return returncode
        mock_proc.poll = poll_shutdown
    else:
        mock_proc.poll = MagicMock(return_value=returncode)
    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read.return_value = stdout
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read.return_value = stderr
    return mock_proc


class TestDaemonMaxWaves:
    @patch("scripts.run_pipeline.subprocess.Popen")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_max_waves_0(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Daemon with max_waves=0 and no pending mandates just sleeps."""
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()

        # Make sleep set shutdown_requested after 1 call to avoid infinite loop
        def sleep_then_stop(*args, **kwargs):
            rp.shutdown_requested = True
        mock_sleep.side_effect = sleep_then_stop

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=0, parallel=1, sleep_seconds=1, mandates_dir=mandates_dir)

        # Should have slept at least once (no mandates found)
        assert mock_sleep.called

    @patch("scripts.run_pipeline.subprocess.Popen")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_max_waves_1(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Daemon with max_waves=1 runs 1 wave and exits."""
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()
        (mandates_dir / "test.json").write_text('{"name": "test"}')

        mock_popen.return_value = _make_fake_popen(returncode=0)

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=1, parallel=1, sleep_seconds=1, mandates_dir=mandates_dir)

        # Should have launched exactly one subprocess
        assert mock_popen.call_count == 1
        cmd_list = mock_popen.call_args[0][0]
        assert any("test.json" in str(x) for x in cmd_list)

    @patch("scripts.run_pipeline.subprocess.Popen")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_graceful_shutdown(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Daemon exits cleanly when shutdown_requested is set mid-wave.

        With parallel=2, both mandates in the batch are launched before the
        first poll sees shutdown_requested.  The key assertion is that the
        daemon stops after the current wave instead of starting a new one.
        """
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()
        (mandates_dir / "a.json").write_text('{"name": "a"}')
        (mandates_dir / "b.json").write_text('{"name": "b"}')

        # All processes complete immediately; poll triggers shutdown
        mock_popen.return_value = _make_fake_popen(returncode=0, trigger_shutdown=True)

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=0, parallel=2, sleep_seconds=60, mandates_dir=mandates_dir)

        # Both mandates were launched in the same parallel batch
        assert mock_popen.call_count == 2
        # Verify state recorded both completions
        state = rp.DaemonState.load(str(rp.STATE_FILE))
        assert len(state.completed_mandates) == 2

    @patch("scripts.run_pipeline.subprocess.Popen")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_parallel_launches(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Daemon with parallel=2 launches up to 2 mandates concurrently."""
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()
        (mandates_dir / "x.json").write_text('{"name": "x"}')
        (mandates_dir / "y.json").write_text('{"name": "y"}')
        (mandates_dir / "z.json").write_text('{"name": "z"}')

        # All processes complete immediately
        mock_popen.return_value = _make_fake_popen(returncode=0)

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=1, parallel=2, sleep_seconds=1, mandates_dir=mandates_dir)

        # parallel=2 with 3 mandates: first wave launches 2, then max_waves hit
        assert mock_popen.call_count == 2

    @patch("scripts.run_pipeline.subprocess.Popen")
    @patch("scripts.run_pipeline.time.sleep")
    def test_daemon_one_failure_doesnt_kill_others(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """One mandate failing doesn't prevent others from completing."""
        _use_tmp_pid(tmp_path)
        rp.shutdown_requested = False  # Reset global flag
        mandates_dir = tmp_path / "mandates"
        mandates_dir.mkdir()
        (mandates_dir / "fail.json").write_text('{"name": "fail"}')
        (mandates_dir / "ok.json").write_text('{"name": "ok"}')

        popen_count = 0
        def fake_popen(*args, **kwargs):
            nonlocal popen_count
            popen_count += 1
            # First process fails, second succeeds
            rc = 1 if popen_count == 1 else 0
            return _make_fake_popen(returncode=rc, stderr="boom" if rc else "")

        mock_popen.side_effect = fake_popen

        with patch("scripts.run_pipeline.acquire_pid", return_value=True), \
             patch("scripts.run_pipeline.release_pid"):
            rp.run_daemon(max_waves=1, parallel=2, sleep_seconds=1, mandates_dir=mandates_dir)

        # Both should have been launched
        assert mock_popen.call_count == 2
        # Check state: 1 success, 1 failure
        state = rp.DaemonState.load(str(rp.STATE_FILE))
        assert len(state.completed_mandates) == 1
        assert len(state.failed_mandates) == 1
