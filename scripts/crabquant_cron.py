#!/usr/bin/env python3
"""CrabQuant Cron — Autonomous wave execution for scheduled refinement.

This script provides the cron job entry point for running refinement waves
autonomously. It discovers pending mandates, executes them in subprocesses,
and collects results.

Usage:
    python scripts/crabquant_cron.py --mandates-dir refinement/mandates/ --runs-dir refinement_runs/
    python scripts/crabquant_cron.py --wave-only
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Default directories
MANDATES_DIR = str(project_root / "refinement" / "mandates")
CRON_RUNS_DIR = str(project_root / "refinement_runs")

# Age threshold in hours — runs older than this with status='running' are stale
STALE_THRESHOLD_HOURS = 4


# ── Mandate Discovery ─────────────────────────────────────────────────────────

def parse_mandate_configs(mandates_dir: str | Path) -> list[dict]:
    """Parse all mandate JSON files from a directory.

    Args:
        mandates_dir: Directory containing .json mandate files.

    Returns:
        List of parsed mandate dicts.
    """
    mandates_path = Path(mandates_dir)
    if not mandates_path.is_dir():
        return []

    mandates: list[dict] = []
    # Skip known non-mandate JSON files
    _skip_names = {"state.json", "lock.json", "report.json"}
    for json_file in sorted(mandates_path.glob("*.json")):
        if json_file.name in _skip_names:
            continue
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, dict):
                mandates.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", json_file.name, e)

    return mandates


def find_pending_mandates(
    mandates_dir: str | Path,
    runs_dir: str | Path,
) -> list[dict]:
    """Find mandates that haven't been run or have stale/failed runs.

    A mandate is pending if:
    - No run directory exists for it, OR
    - Its most recent run has status='running' but is older than STALE_THRESHOLD_HOURS, OR
    - Its most recent run has status='error' or 'failed'

    Args:
        mandates_dir: Directory of mandate JSON files.
        runs_dir: Directory of run result directories.

    Returns:
        List of mandate dicts that are pending execution.
    """
    all_mandates = parse_mandate_configs(mandates_dir)
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        # No runs at all — everything is pending
        return all_mandates

    # Build a map: mandate_name -> list of state.json contents
    mandate_runs: dict[str, list[dict]] = {}
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        state_file = run_dir / "state.json"
        if not state_file.exists():
            continue
        try:
            state = json.loads(state_file.read_text())
            name = state.get("mandate_name", "")
            if name:
                mandate_runs.setdefault(name, []).append(state)
        except (json.JSONDecodeError, OSError):
            continue

    now = datetime.now(timezone.utc)

    pending: list[dict] = []
    for mandate in all_mandates:
        name = mandate.get("name", "")
        runs = mandate_runs.get(name, [])

        if not runs:
            # Never run
            pending.append(mandate)
            continue

        # Check most recent run
        latest = runs[-1]
        status = latest.get("status", "")

        if status == "success":
            # Already completed successfully
            continue

        if status == "running":
            # Check if stale
            created_at = latest.get("created_at", "")
            try:
                run_time = datetime.fromisoformat(created_at)
                age_hours = (now - run_time).total_seconds() / 3600
                if age_hours > STALE_THRESHOLD_HOURS:
                    pending.append(mandate)
            except (ValueError, TypeError):
                # Can't parse time — treat as stale
                pending.append(mandate)
            continue

        # Error, failed, max_turns, abandoned — retry
        pending.append(mandate)

    return pending


# ── Single Mandate Execution ──────────────────────────────────────────────────

def run_single_mandate(
    mandate_path: str,
    runs_dir: str | Path | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Execute a single mandate by calling the refinement loop in a subprocess.

    Args:
        mandate_path: Path to the mandate JSON file.
        runs_dir: Directory for run output. Defaults to CRON_RUNS_DIR.
        timeout: Subprocess timeout in seconds.

    Returns:
        Dict with status, mandate name, and any error info.
    """
    runs_dir = runs_dir or CRON_RUNS_DIR

    script_path = project_root / "scripts" / "refinement_loop.py"

    # Load mandate to pass its config to the subprocess
    with open(mandate_path) as f:
        mandate_cfg = json.load(f)

    cmd = [
        sys.executable,
        str(script_path),
        "--mandate", mandate_path,
        "--sharpe-target", str(mandate_cfg.get("sharpe_target", 1.5)),
        "--max-turns", str(mandate_cfg.get("max_turns", 5)),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
        )

        # Try to parse the output for status info
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Extract run directory from output
        run_dir = ""
        for line in stdout.splitlines():
            if "Run directory:" in line or "run_" in line.lower():
                # Best effort extraction
                run_dir = line.strip()

        return {
            "mandate": mandate_path,
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": stdout[-500:],  # Truncate
            "stderr": stderr[-500:],
            "run_dir": run_dir,
            "error": stderr[:200] if result.returncode != 0 else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "mandate": mandate_path,
            "status": "error",
            "error": f"Timeout after {timeout}s",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "mandate": mandate_path,
            "status": "error",
            "error": str(e),
            "returncode": -1,
        }


# ── Wave Execution ────────────────────────────────────────────────────────────

def run_wave(
    mandates: list[dict],
    max_parallel: int = 3,
    timeout: int = 300,
) -> dict[str, Any]:
    """Execute a wave of mandates.

    Each mandate is written to a temp JSON file and executed in a subprocess.
    Uses threading for parallelism (subprocess does the heavy lifting).

    Args:
        mandates: List of mandate dicts.
        max_parallel: Maximum concurrent subprocesses.
        timeout: Per-mandate timeout in seconds.

    Returns:
        Dict with total, successful, failed, and per-mandate results.
    """
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for mandate in mandates:
            # Write mandate to temp file
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="cron_mandate_"
            )
            json.dump(mandate, tmp)
            tmp.close()

            future = executor.submit(
                run_single_mandate, tmp.name, timeout=timeout
            )
            futures[future] = tmp.name

        for future in as_completed(futures):
            tmp_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "mandate": tmp_path,
                    "status": "error",
                    "error": str(e),
                })
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") != "success")

    return {
        "total": len(mandates),
        "successful": successful,
        "failed": failed,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Result Collection ─────────────────────────────────────────────────────────

def collect_wave_results(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Collect results from all run directories.

    Reads state.json from each subdirectory and extracts key metrics.

    Args:
        runs_dir: Directory containing run result subdirectories.

    Returns:
        List of result dicts with mandate, status, sharpe, turn, etc.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        state_file = run_dir / "state.json"
        if not state_file.exists():
            continue

        try:
            state = json.loads(state_file.read_text())
            results.append({
                "mandate": state.get("mandate_name", ""),
                "status": state.get("status", "unknown"),
                "best_sharpe": state.get("best_sharpe", 0.0),
                "current_turn": state.get("current_turn", 0),
                "run_id": state.get("run_id", ""),
                "created_at": state.get("created_at", ""),
                "history": state.get("history", []),
                "run_dir": str(run_dir),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """Command line entry point for cron execution."""
    import argparse

    parser = argparse.ArgumentParser(description="CrabQuant Cron — Autonomous Wave Execution")
    parser.add_argument(
        "--mandates-dir", default=MANDATES_DIR,
        help="Directory containing mandate JSON files",
    )
    parser.add_argument(
        "--runs-dir", default=CRON_RUNS_DIR,
        help="Directory for run output",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=3,
        help="Maximum concurrent mandate runs",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-mandate timeout in seconds",
    )
    parser.add_argument(
        "--wave-only", action="store_true",
        help="Run all mandates in the mandates-dir (skip pending check)",
    )

    args = parser.parse_args()

    # Discover mandates
    if args.wave_only:
        mandates = parse_mandate_configs(args.mandates_dir)
    else:
        mandates = find_pending_mandates(args.mandates_dir, args.runs_dir)

    if not mandates:
        print("No pending mandates found.")
        return

    print(f"Found {len(mandates)} mandates to run.")

    # Execute wave
    report = run_wave(
        mandates,
        max_parallel=args.max_parallel,
        timeout=args.timeout,
    )

    print(f"\nWave complete: {report['successful']}/{report['total']} successful")
    print(f"Timestamp: {report['timestamp']}")

    # Collect and display results
    results = collect_wave_results(args.runs_dir)
    if results:
        print(f"\nTotal runs on record: {len(results)}")
        for r in results:
            print(f"  {r['mandate']}: {r['status']} (Sharpe: {r['best_sharpe']:.2f})")


if __name__ == "__main__":
    main()
