#!/usr/bin/env python3
"""
Wave Runner CLI — Launch parallel refinement waves.

Usage:
    python scripts/wave_runner.py --mandates refinement/mandates/ --parallel 3 --wave-size 5
    python scripts/wave_runner.py --mandates refinement/mandates/ --parallel 5 --max-waves 20
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crabquant.refinement.wave_manager import run_waves


def main():
    parser = argparse.ArgumentParser(description="Run refinement waves")
    parser.add_argument("--mandates", required=True, help="Directory of mandate JSON files")
    parser.add_argument("--parallel", type=int, default=3, help="Max concurrent mandates")
    parser.add_argument("--wave-size", type=int, default=5, help="Mandates per wave")
    parser.add_argument("--max-waves", type=int, default=10, help="Max waves to run")
    parser.add_argument("--stop-convergence", type=float, default=0.6, help="Stop if convergence exceeds this")
    args = parser.parse_args()

    reports = run_waves(
        mandate_dir=args.mandates,
        max_parallel=args.parallel,
        wave_size=args.wave_size,
        max_waves=args.max_waves,
        stop_on_convergence=args.stop_convergence,
    )

    # Summary
    total_success = sum(r.successful for r in reports)
    total_runs = sum(r.total_mandates for r in reports)
    print(f"\n{'=' * 60}")
    print(f"Total: {total_success}/{total_runs} strategies converged ({total_success / max(total_runs, 1):.0%})")
    print(f"Waves: {len(reports)}")


if __name__ == "__main__":
    main()
