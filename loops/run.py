#!/usr/bin/env python3
"""Loop Runner CLI — discover, validate, and launch agent experiment loops.

Usage:
    python loops/run.py list                        # list all available loops
    python loops/run.py diversity-explorer          # print program.md
    python loops/run.py diversity-explorer --dry-run  # validate loop config
    python loops/run.py diversity-explorer --report  # print experiment summary

This is NOT an autonomous runner — it prepares context for an AI agent to read
program.md and execute the loop.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

LOOPS_DIR = Path(__file__).resolve().parent


def discover_loops() -> list[str]:
    """Find all loop names registered in registry.yaml.

    Returns:
        Sorted list of loop names.
    """
    registry = load_registry()
    return sorted(registry.keys())


def load_registry() -> dict:
    """Load registry.yaml and return the loops dict.

    Returns:
        Dict of loop metadata, or empty dict if registry doesn't exist.
    """
    registry_path = LOOPS_DIR / "registry.yaml"
    if not registry_path.exists():
        return {}
    try:
        import yaml
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        return data.get("loops", {}) if data else {}
    except Exception:
        return {}


def load_program_md(loop_name: str) -> str:
    """Read the full program.md for a loop.

    Args:
        loop_name: Name of the loop directory.

    Returns:
        Content of program.md as string.

    Raises:
        FileNotFoundError: If program.md doesn't exist.
    """
    path = LOOPS_DIR / loop_name / "program.md"
    if not path.exists():
        raise FileNotFoundError(f"No program.md for loop '{loop_name}' at {path}")
    return path.read_text()


def get_tsv_path(loop_name: str) -> Path | None:
    """Resolve the TSV log path for a loop from registry.

    Args:
        loop_name: Name of the loop.

    Returns:
        Path to the TSV log file, or None if not configured.
    """
    registry = load_registry()
    meta = registry.get(loop_name, {})
    tsv_log = meta.get("tsv_log")
    if not tsv_log:
        return None
    return LOOPS_DIR / tsv_log


def validate_loop(loop_name: str) -> tuple[bool, list[str]]:
    """Validate a loop's configuration files.

    Checks:
        - Loop directory exists
        - program.md exists and is non-empty
        - TSV log (if configured) has valid headers
        - Loop is registered in registry.yaml

    Args:
        loop_name: Name of the loop directory.

    Returns:
        Tuple of (is_valid, list_of_issues). Empty issues list = valid.
    """
    loop_dir = LOOPS_DIR / loop_name
    issues = []

    # Check registry
    registry = load_registry()
    if loop_name not in registry:
        issues.append(f"Loop '{loop_name}' not found in registry.yaml")

    if not loop_dir.exists():
        return False, [f"Loop directory '{loop_name}' does not exist"]

    program_path = loop_dir / "program.md"
    if not program_path.exists():
        issues.append("program.md is missing")
    elif program_path.stat().st_size == 0:
        issues.append("program.md is empty")

    # Check TSV log if configured
    tsv_path = get_tsv_path(loop_name)
    if tsv_path and tsv_path.exists():
        try:
            content = tsv_path.read_text()
            lines = [l for l in content.splitlines() if l.strip()]
            if len(lines) < 1:
                issues.append(f"{tsv_path.name} exists but is empty (no header)")
            else:
                # Validate header row has tabs (not just spaces)
                header = lines[0]
                if "\t" not in header:
                    issues.append(f"{tsv_path.name} header is not valid TSV (no tab separators)")
                else:
                    cols = header.split("\t")
                    if len(cols) < 2:
                        issues.append(f"{tsv_path.name} header has only {len(cols)} column(s), expected at least 2")
        except Exception as e:
            issues.append(f"{tsv_path.name} is not readable: {e}")

    return len(issues) == 0, issues


def generate_report(loop_name: str) -> str:
    """Read a loop's TSV log and compute summary statistics.

    Args:
        loop_name: Name of the loop.

    Returns:
        Formatted report string.
    """
    tsv_path = get_tsv_path(loop_name)
    if not tsv_path:
        return f"No TSV log configured for '{loop_name}'."

    if not tsv_path.exists():
        return f"No {tsv_path.name} found for '{loop_name}' — no experiments run yet."

    total = 0
    statuses: dict[str, int] = {}
    last_entry = ""
    metric_values: list[float] = []
    status_col_idx = None
    metric_col_idx = None
    description_col_idx = None

    with open(tsv_path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return f"{tsv_path.name} is empty (no header)."

        # Try to find metric/status/description columns
        for i, col in enumerate(header):
            col_lower = col.lower()
            if col_lower in ("status", "result_status"):
                status_col_idx = i
            if col_lower in ("metric", "result_sharpe", "sharpe_ratio"):
                metric_col_idx = i
            if col_lower in ("description", "gap_description", "strategy_name", "insight"):
                description_col_idx = i

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            total += 1

            # Track status
            if status_col_idx is not None and status_col_idx < len(row):
                status = row[status_col_idx].strip()
                statuses[status] = statuses.get(status, 0) + 1

            # Track metric value
            if metric_col_idx is not None and metric_col_idx < len(row):
                try:
                    metric_values.append(float(row[metric_col_idx]))
                except (ValueError, TypeError):
                    pass

            # Track last entry description
            if description_col_idx is not None and description_col_idx < len(row):
                last_entry = row[description_col_idx].strip()
            elif row:
                last_entry = row[-1].strip()[:80]

    if total == 0:
        return f"No data rows in {tsv_path.name}."

    lines = [
        f"{'=' * 50}",
        f"  Experiment Report: {loop_name}",
        f"{'=' * 50}",
        f"  Total entries:     {total}",
    ]

    if statuses:
        for status, count in sorted(statuses.items()):
            pct = count / total * 100
            lines.append(f"  {status:20s} {count:4d}  ({pct:.1f}%)")

    if metric_values:
        best = max(metric_values)
        avg = sum(metric_values) / len(metric_values)
        worst = min(metric_values)
        lines.append(f"  Best metric:       {best:.6f}")
        lines.append(f"  Avg metric:        {avg:.6f}")
        lines.append(f"  Worst metric:      {worst:.6f}")
        if len(metric_values) > 1:
            improvement = ((best - metric_values[0]) / abs(metric_values[0]) * 100) if metric_values[0] != 0 else 0.0
            lines.append(f"  Improvement:       {improvement:+.2f}% (from first entry)")

    if last_entry:
        lines.append(f"  Last entry:        {last_entry[:80]}")

    lines.append(f"{'=' * 50}")
    return "\n".join(lines)


def cmd_list(args):
    """List all available loops with summaries."""
    registry = load_registry()

    if not registry:
        print("No loops found in registry.yaml.")
        return 1

    print(f"Available loops ({len(registry)}):\n")
    for name in sorted(registry.keys()):
        meta = registry[name]
        desc = meta.get("description", "No description")
        category = meta.get("category", "unknown")
        metric = meta.get("metric", "unknown")
        direction = meta.get("direction", "unknown")
        print(f"  {name}")
        print(f"    Description: {desc}")
        print(f"    Category:    {category}")
        print(f"    Metric:      {metric} ({direction})")
        print()
    return 0


def cmd_run(args):
    """Print program.md and prepare context for an agent."""
    loop_name = args.loop_name

    # Validate the loop exists
    try:
        content = load_program_md(loop_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Print summary from registry
    registry = load_registry()
    meta = registry.get(loop_name, {})
    if meta:
        print(f"Loop: {loop_name}")
        print(f"Description: {meta.get('description', 'N/A')}")
        print(f"Category:    {meta.get('category', 'N/A')}")
        print(f"Metric:      {meta.get('metric', 'N/A')} ({meta.get('direction', 'N/A')})")
        print()

    # Print the full program.md
    print(f"{'=' * 60}")
    print(f"  program.md — {loop_name}")
    print(f"{'=' * 60}")
    print()
    print(content)
    print()
    print(f"{'=' * 60}")
    print("  An AI agent should read the above program.md and execute the loop.")
    print(f"{'=' * 60}")
    return 0


def cmd_dry_run(args):
    """Validate a loop's configuration without running anything."""
    loop_name = args.loop_name

    valid, issues = validate_loop(loop_name)
    if valid:
        print(f"✅ Loop '{loop_name}' is valid.")
        print(f"   program.md:  OK")

        tsv_path = get_tsv_path(loop_name)
        if tsv_path and tsv_path.exists():
            count = sum(1 for l in tsv_path.read_text().splitlines() if l.strip()) - 1
            print(f"   {tsv_path.name}: {max(count, 0)} entries recorded")
        elif tsv_path:
            print(f"   {tsv_path.name}: not yet created (will be created on first run)")
        else:
            print(f"   TSV log:      not configured")
    else:
        print(f"❌ Loop '{loop_name}' has issues:")
        for issue in issues:
            print(f"   - {issue}")
        return 1
    return 0


def cmd_report(args):
    """Print experiment report for a loop."""
    loop_name = args.loop_name
    report = generate_report(loop_name)
    print(report)
    return 0


def main():
    # Pre-process sys.argv to handle shorthand forms:
    #   python run.py list                         → python run.py list
    #   python run.py diversity-explorer            → python run.py run diversity-explorer
    #   python run.py diversity-explorer --dry-run  → python run.py dry-run diversity-explorer
    #   python run.py diversity-explorer --report   → python run.py report diversity-explorer
    if len(sys.argv) > 1 and sys.argv[1] not in ("list", "run", "dry-run", "report", "-h", "--help"):
        first_arg = sys.argv[1]
        remaining = sys.argv[2:]
        if "--dry-run" in remaining:
            remaining = [a for a in remaining if a != "--dry-run"]
            sys.argv = [sys.argv[0], "dry-run", first_arg] + remaining
        elif "--report" in remaining:
            remaining = [a for a in remaining if a != "--report"]
            sys.argv = [sys.argv[0], "report", first_arg] + remaining
        else:
            sys.argv = [sys.argv[0], "run", first_arg] + remaining

    parser = argparse.ArgumentParser(
        description="Loop Runner — discover and launch agent experiment loops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python loops/run.py list                         List all loops
  python loops/run.py diversity-explorer           Print program.md
  python loops/run.py diversity-explorer --dry-run Validate config
  python loops/run.py diversity-explorer --report  Show experiment stats
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    subparsers.add_parser("list", help="List all available loops")

    # run command (default: loop name)
    run_parser = subparsers.add_parser("run", help="Prepare context for an agent to run a loop")
    run_parser.add_argument("loop_name", help="Name of the loop to run")

    # dry-run command
    dry_parser = subparsers.add_parser("dry-run", help="Validate loop config without running")
    dry_parser.add_argument("loop_name", help="Name of the loop to validate")

    # report command
    report_parser = subparsers.add_parser("report", help="Print experiment summary")
    report_parser.add_argument("loop_name", help="Name of the loop")

    args = parser.parse_args()

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "dry-run":
        return cmd_dry_run(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
