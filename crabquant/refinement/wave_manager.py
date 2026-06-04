"""
Wave Manager - Parallel mandate execution via subprocess isolation.

Each mandate runs in its own subprocess to avoid shared state issues:
- STRATEGY_REGISTRY, indicator_cache, sys.modules are all per-process
- Run directories are unique (timestamped)
- Winners file uses file locking

See PRD §23 for full specification.
"""

import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class WaveResult:
    """Result of a single mandate run within a wave."""
    mandate_name: str
    status: str              # success | max_turns | stuck | failed | error | abandoned
    best_sharpe: float
    turns_used: int
    run_dir: str
    error: Optional[str] = None


@dataclass
class WaveReport:
    """Summary of an entire wave."""
    wave_number: int
    started_at: str
    completed_at: str
    total_mandates: int
    successful: int
    failed: int
    results: List[WaveResult] = field(default_factory=list)

    @property
    def convergence_rate(self) -> float:
        return self.successful / max(self.total_mandates, 1)

    def to_dict(self) -> dict:
        return {
            "wave_number": self.wave_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_mandates": self.total_mandates,
            "successful": self.successful,
            "failed": self.failed,
            "convergence_rate": self.convergence_rate,
            "results": [
                {
                    "mandate": r.mandate_name,
                    "status": r.status,
                    "sharpe": r.best_sharpe,
                    "turns": r.turns_used,
                    "run_dir": r.run_dir,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def run_single_mandate(mandate_path: str, extra_args: Optional[List[str]] = None) -> WaveResult:
    """Run a single mandate in a subprocess. Returns WaveResult."""
    mandate_path = str(mandate_path)
    mandate = json.loads(Path(mandate_path).read_text())
    mandate_name = mandate.get("name", Path(mandate_path).stem)

    # Resolve the refinement_loop.py script path
    script_dir = Path(__file__).parent.parent.parent
    script_path = script_dir / "scripts" / "refinement_loop.py"

    cmd = ["python", str(script_path), "--mandate", mandate_path]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per mandate
            cwd=str(script_dir),
        )

        # Parse state.json from the run directory created by refinement_loop
        # The run dir is logged by refinement_loop or we can find it by timestamp
        runs_dir = script_dir / "refinement_runs"
        if runs_dir.exists():
            # Find the most recent run directory for this mandate
            matching_dirs = sorted(
                runs_dir.glob(f"{mandate_name.replace(' ', '_').lower()}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for run_dir in matching_dirs[:3]:
                state_path = run_dir / "state.json"
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    return WaveResult(
                        mandate_name=mandate_name,
                        status=state.get("status", "unknown"),
                        best_sharpe=state.get("best_sharpe", 0),
                        turns_used=state.get("current_turn", 0),
                        run_dir=str(run_dir),
                    )

        return WaveResult(
            mandate_name=mandate_name,
            status="error",
            best_sharpe=0,
            turns_used=0,
            run_dir="",
            error=result.stderr[-500:] if result.stderr else "No run directory found",
        )

    except subprocess.TimeoutExpired:
        return WaveResult(
            mandate_name=mandate_name,
            status="error",
            best_sharpe=0,
            turns_used=0,
            run_dir="",
            error="Subprocess timed out (600s)",
        )
    except Exception as e:
        return WaveResult(
            mandate_name=mandate_name,
            status="error",
            best_sharpe=0,
            turns_used=0,
            run_dir="",
            error=str(e),
        )


def run_wave(mandate_paths: List[str], max_parallel: int = 3) -> WaveReport:
    """Run a wave of mandates in parallel.

    Args:
        mandate_paths: List of paths to mandate JSON files
        max_parallel: Max concurrent subprocesses

    Returns:
        WaveReport with results for each mandate
    """
    report = WaveReport(
        wave_number=0,  # set by caller
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at="",
        total_mandates=len(mandate_paths),
        successful=0,
        failed=0,
    )

    print(f"🌊 Wave starting: {len(mandate_paths)} mandates, {max_parallel} parallel")

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_single_mandate, path): path
            for path in mandate_paths
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = WaveResult(
                    mandate_name=Path(path).stem,
                    status="error",
                    best_sharpe=0,
                    turns_used=0,
                    run_dir="",
                    error=str(e),
                )

            report.results.append(result)
            if result.status == "success":
                report.successful += 1
                print(f"  ✅ {result.mandate_name}: Sharpe {result.best_sharpe:.2f} ({result.turns_used} turns)")
            else:
                report.failed += 1
                print(f"  ❌ {result.mandate_name}: {result.status} ({result.error or 'did not converge'})")

    report.completed_at = datetime.now(timezone.utc).isoformat()
    print(f"\n🌊 Wave complete: {report.successful}/{report.total_mandates} converged ({report.convergence_rate:.0%})")
    return report


def run_waves(
    mandate_dir: str,
    max_parallel: int = 3,
    wave_size: int = 5,
    max_waves: int = 10,
    stop_on_convergence: float = 0.6,
) -> List[WaveReport]:
    """Run multiple waves until convergence rate exceeds target or max waves hit.

    Args:
        mandate_dir: Directory of mandate JSON files
        max_parallel: Max concurrent mandates per wave
        wave_size: Mandates per wave
        max_waves: Stop after this many waves
        stop_on_convergence: Stop if convergence rate exceeds this (e.g., 0.6 = 60%)

    Returns:
        List of WaveReport for each wave executed
    """
    mandate_paths = sorted(Path(mandate_dir).glob("*.json"))
    if not mandate_paths:
        print(f"No mandate files found in {mandate_dir}")
        return []

    all_reports: List[WaveReport] = []
    runs_dir = Path(mandate_dir).parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for wave_num in range(1, max_waves + 1):
        # Pick mandates for this wave (cycle through available)
        start_idx = (wave_num - 1) * wave_size % len(mandate_paths)
        wave_paths = []
        for i in range(wave_size):
            wave_paths.append(str(mandate_paths[(start_idx + i) % len(mandate_paths)]))

        report = run_wave(wave_paths, max_parallel)
        report.wave_number = wave_num
        all_reports.append(report)

        # Save wave report
        report_path = runs_dir / f"wave_{wave_num:03d}_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

        # Check stopping condition
        if report.convergence_rate >= stop_on_convergence:
            print(f"\n🎯 Convergence target met: {report.convergence_rate:.0%} >= {stop_on_convergence:.0%}")
            break

        # Brief pause between waves to let API rate limits reset.
        # 10s is conservative but ensures the rolling 5h window advances.
        if wave_num < max_waves:
            time.sleep(10)
    
    return all_reports
