#!/usr/bin/env python3
"""
CrabQuant Pipeline Daemon

Continuous loop: pick mandates → run them → promote winners → repeat.
Supports foreground, daemon (background), stop, and status modes.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crabquant.refinement.state import DaemonState

PID_FILE = project_root / "crabquant.pid"
STATE_FILE = project_root / "results" / "daemon_state.json"
HEARTBEAT_FILE = project_root / "results" / "daemon_heartbeat.json"
DEFAULT_MANDATES_DIR = project_root / "mandates"
DEFAULT_PARALLEL = 3
DEFAULT_SLEEP = 60

shutdown_requested = False


def handle_signal(signum, frame):
    """Set global flag so the daemon loop can exit cleanly."""
    global shutdown_requested
    shutdown_requested = True


# ── PID management ────────────────────────────────────────────────────────


def acquire_pid() -> bool:
    """Write PID file. Return True if successful, False if already running."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)  # Check if alive
            return False  # Already running
        except (ProcessLookupError, ValueError):
            pass  # Stale PID, take over
    PID_FILE.write_text(str(os.getpid()))
    return True


def release_pid() -> None:
    """Remove PID file if it belongs to us."""
    if PID_FILE.exists():
        try:
            if int(PID_FILE.read_text().strip()) == os.getpid():
                PID_FILE.unlink()
        except ValueError:
            pass


# ── Stop / Status ─────────────────────────────────────────────────────────


def stop_daemon() -> None:
    """Read PID, send SIGTERM, wait for clean exit."""
    if not PID_FILE.exists():
        print("No daemon running (no PID file)")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to daemon (PID {pid})")
        # Wait up to 10 seconds for clean exit
        for _ in range(10):
            time.sleep(1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print("Daemon stopped cleanly")
                return
        print("Daemon did not stop in 10s — may need manual kill")
    except ProcessLookupError:
        print("Daemon not running")
        release_pid()


def print_status() -> None:
    """Print daemon status."""
    alive = False
    pid = None
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            alive = True
        except (ProcessLookupError, ValueError):
            pass

    state = DaemonState.load(str(STATE_FILE))

    print(f"Daemon: {'RUNNING' if alive else 'STOPPED'}")
    if pid:
        print(f"PID: {pid}")
    if state:
        print(f"Wave: {state.current_wave}")
        print(f"Mandates run: {state.total_mandates_run}")
        print(f"Promoted: {state.total_strategies_promoted}")
        print(f"Last heartbeat: {state.last_heartbeat}")
        print(f"Pending: {len(state.pending_mandates)}")
    else:
        print("No state file found")


# ── Mandate discovery ─────────────────────────────────────────────────────


def discover_mandates(mandates_dir: Path) -> list:
    """Find all .json mandate files in directory."""
    if not mandates_dir.exists():
        return []
    return sorted(f.name for f in mandates_dir.glob("*.json"))


# ── Subprocess runner ─────────────────────────────────────────────────────


def run_single_mandate(
    mandate_path: str,
    max_turns: int = 7,
    sharpe_target: float = 1.5,
    timeout: int = 600,
) -> dict:
    """Run a single mandate via subprocess. Returns result dict."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/refinement_loop.py",
                "--mandate",
                mandate_path,
                "--max-turns",
                str(max_turns),
                "--sharpe-target",
                str(sharpe_target),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"returncode": -2, "stdout": "", "stderr": str(e)}


# ── Daemon loop ───────────────────────────────────────────────────────────


def run_daemon(
    max_waves: int,
    parallel: int,
    sleep_seconds: int,
    mandates_dir: Path,
) -> None:
    """Main daemon loop — runs waves until max_waves or shutdown signal."""
    global shutdown_requested

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if not acquire_pid():
        print("Daemon already running. Use --stop or --status.")
        sys.exit(1)

    print(f"CrabQuant daemon started (PID {os.getpid()})")
    print(f"Max waves: {'unlimited' if max_waves == 0 else max_waves}")
    print(f"Parallel: {parallel}, Sleep: {sleep_seconds}s")
    print(f"Mandates dir: {mandates_dir}")

    # Load or create state
    state = DaemonState.load(str(STATE_FILE))
    if state is None:
        state = DaemonState.create()
        state.pending_mandates = discover_mandates(mandates_dir)
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state.save(str(STATE_FILE))
        print(f"Fresh state created with {len(state.pending_mandates)} mandates")
    else:
        print(
            f"Resumed from saved state (wave {state.current_wave}, "
            f"{len(state.pending_mandates)} pending)"
        )

    try:
        wave = state.current_wave
        while not shutdown_requested:
            # ── Ensure we have pending mandates ───────────────────────
            if not state.pending_mandates:
                if max_waves > 0 and wave >= max_waves:
                    print(f"Max waves ({max_waves}) reached. Exiting.")
                    break
                all_mandates = discover_mandates(mandates_dir)
                completed_or_failed = set(state.completed_mandates + state.failed_mandates)
                new_mandates = [m for m in all_mandates if m not in completed_or_failed]
                if new_mandates:
                    state.pending_mandates = new_mandates
                    state.save(str(STATE_FILE))
                    print(f"Discovered {len(new_mandates)} new mandates")
                else:
                    print(f"No new mandates found. Sleeping {sleep_seconds}s...")
                    time.sleep(sleep_seconds)
                    state.heartbeat(str(STATE_FILE))
                    continue

            # ── Pick mandates for this wave ───────────────────────────
            wave_mandates = state.pending_mandates[:parallel]
            wave += 1
            state.current_wave = wave
            # record_wave_start requires a mandate_name — use the first one
            state.record_wave_start(wave, wave_mandates[0], str(STATE_FILE))

            print(f"\n{'=' * 60}")
            print(f"Wave {wave} — running {len(wave_mandates)} mandates")

            # ── Run mandates in parallel (subprocess isolation) ───────
            active_procs: list[dict] = []  # {proc, name, stdout, stderr}
            wave_results: list[dict] = []
            started: set[str] = set()
            idx = 0

            while idx < len(wave_mandates) or active_procs:
                if shutdown_requested:
                    break

                # Fill slots up to --parallel concurrency
                while (idx < len(wave_mandates)
                       and len(active_procs) < parallel):
                    mandate_name = wave_mandates[idx]
                    idx += 1
                    mandate_path = str(mandates_dir / mandate_name)
                    print(f"  ▶ Starting: {mandate_name}", flush=True)
                    try:
                        proc = subprocess.Popen(
                            [
                                sys.executable,
                                "scripts/refinement_loop.py",
                                "--mandate",
                                mandate_path,
                                "--max-turns",
                                str(7),
                                "--sharpe-target",
                                str(1.5),
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=str(project_root),
                        )
                        active_procs.append({
                            "proc": proc,
                            "name": mandate_name,
                            "stdout": "",
                            "stderr": "",
                        })
                        started.add(mandate_name)
                    except Exception as exc:
                        print(f"  ❌ Launch failed: {mandate_name} — {exc}")
                        wave_results.append({
                            "name": mandate_name,
                            "returncode": -2,
                            "stdout": "",
                            "stderr": str(exc),
                        })

                # Poll finished processes (non-blocking)
                still_running = []
                for entry in active_procs:
                    ret = entry["proc"].poll()
                    if ret is None:
                        still_running.append(entry)
                    else:
                        # Process finished — collect output
                        stdout_data = entry["proc"].stdout.read() or ""
                        stderr_data = entry["proc"].stderr.read() or ""
                        wave_results.append({
                            "name": entry["name"],
                            "returncode": ret,
                            "stdout": stdout_data[-2000:],
                            "stderr": stderr_data[-1000:],
                        })
                        entry["proc"].stdout.close()
                        entry["proc"].stderr.close()
                active_procs = still_running

                # Brief sleep to avoid busy-looping when processes are running
                if active_procs:
                    time.sleep(0.5)

            # ── Record results for completed mandates ───────────────
            for wr in wave_results:
                if wr["returncode"] == 0:
                    state.record_wave_completion(
                        wr["name"], "success", 0.0, str(STATE_FILE)
                    )
                    print(f"  ✅ Completed: {wr['name']}")
                else:
                    state.record_wave_completion(
                        wr["name"], "failed", 0.0, str(STATE_FILE)
                    )
                    print(f"  ❌ Failed: {wr['name']} — {wr['stderr'][:200]}")

            # ── Post-wave: heartbeat + heartbeat file ─────────────────
            state.last_wave_completed = datetime.now(timezone.utc).isoformat()
            state.heartbeat(str(STATE_FILE))

            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.write_text(
                json.dumps(
                    {
                        "wave": wave,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "mandates_run": state.total_mandates_run,
                        "promoted": state.total_strategies_promoted,
                    },
                    indent=2,
                )
            )

            if max_waves > 0 and wave >= max_waves:
                print(f"Max waves ({max_waves}) reached. Exiting.")
                break

            print(f"Wave {wave} complete. Sleeping {sleep_seconds}s...")
            time.sleep(sleep_seconds)

        # Clean shutdown
        state.mark_shutdown(str(STATE_FILE))
        print("Daemon shutting down cleanly.")

    finally:
        release_pid()


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="CrabQuant Pipeline Daemon")
    parser.add_argument("--daemon", action="store_true", help="Run in background")
    parser.add_argument(
        "--max-waves", type=int, default=0, help="Max waves (0=unlimited)"
    )
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument(
        "--sleep", type=int, default=DEFAULT_SLEEP, help="Seconds between waves"
    )
    parser.add_argument(
        "--mandates-dir", type=str, default=str(DEFAULT_MANDATES_DIR)
    )
    parser.add_argument("--stop", action="store_true", help="Stop running daemon")
    parser.add_argument("--status", action="store_true", help="Print daemon status")
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
        return

    if args.status:
        print_status()
        return

    mandates_dir = Path(args.mandates_dir)

    if args.daemon:
        # Double-fork for background (Linux / WSL2 compatible)
        if os.fork():
            os._exit(0)
        os.setsid()
        if os.fork():
            os._exit(0)

        # Redirect stdout/stderr to log file
        log_file = project_root / "results" / "daemon.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(log_file, "a")
        sys.stderr = open(log_file, "a")

    run_daemon(args.max_waves, args.parallel, args.sleep, mandates_dir)


if __name__ == "__main__":
    main()
