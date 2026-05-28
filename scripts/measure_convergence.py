#!/usr/bin/env python3
"""CrabQuant Convergence Report — standing convergence metric tool.

Parses all refinement_runs/ directories and reports convergence statistics:
overall rate, per-archetype/ticker success, abandonment, failure modes,
promotion rate, and time trends.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWN_ARCHETYPES = [
    "momentum",
    "mean_reversion",
    "breakout",
    "trend",
    "volume",
    "volatility",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrabQuant Convergence Report"
    )
    parser.add_argument(
        "--runs-dir",
        default="refinement_runs",
        help="Directory containing refinement run subdirectories (default: refinement_runs/)",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output results as JSON instead of human-readable table",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only consider runs created after this date (YYYY-MM-DD format)",
    )
    return parser.parse_args(argv)


def load_runs(runs_dir: Path, since: datetime | None) -> list[dict[str, Any]]:
    """Load all state.json files from runs_dir, optionally filtering by date."""
    runs: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return runs

    for subdir in sorted(runs_dir.iterdir()):
        state_path = subdir / "state.json"
        if not state_path.exists():
            continue

        try:
            data = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Filter by date if --since provided
        if since is not None:
            created = data.get("created_at", "")
            if created:
                try:
                    run_dt = datetime.fromisoformat(created)
                except (ValueError, TypeError):
                    continue
                if run_dt < since:
                    continue

        runs.append(data)
    return runs


def extract_archetype(mandate_name: str) -> str | None:
    """Extract archetype from mandate name (e.g. 'momentum_aapl' -> 'momentum')."""
    name_lower = mandate_name.lower().replace(" ", "_")
    for arch in KNOWN_ARCHETYPES:
        if name_lower.startswith(arch + "_") or name_lower.startswith(arch + " "):
            return arch
        # Also check if archetype appears as a standalone word
        parts = name_lower.split("_")
        if parts and parts[0] == arch:
            return arch
    return None


def extract_tickers(data: dict[str, Any]) -> list[str]:
    """Extract ticker list from run data."""
    tickers = data.get("tickers", [])
    if isinstance(tickers, list):
        return [t.upper() for t in tickers]
    return []


def collect_failure_modes(runs: list[dict[str, Any]]) -> Counter[str]:
    """Collect all failure-mode classifications from run histories."""
    modes: Counter[str] = Counter()
    for run in runs:
        for entry in run.get("history", []):
            # History entries may have 'status' (code_generation_failed, circuit_breaker_open)
            # or 'failure_mode' (low_sharpe, high_drawdown, etc.) — use whichever is present
            mode = entry.get("status") or entry.get("failure_mode")
            if mode and mode != "success":
                modes[mode] += 1
    return modes


def compute_report(runs: list[dict[str, Any]], promoted_codes: set[str] | None = None) -> dict[str, Any]:
    """Compute all convergence metrics from loaded runs."""
    total = len(runs)
    if total == 0:
        return {
            "total_mandates": 0,
            "convergence_rate": 0.0,
            "converged": 0,
            "avg_turns_to_converge": 0.0,
            "abandoned": 0,
            "max_turns_exhausted": 0,
            "success": 0,
            "running": 0,
            "by_archetype": {},
            "by_ticker": {},
            "failure_modes": {},
            "promotion_rate": 0.0,
            "time_trends": {},
        }

    success_count = 0
    abandoned_count = 0
    exhausted_count = 0
    running_count = 0
    turns_to_converge: list[int] = []
    converged_codes: set[str] = set()

    # Per-archetype tracking
    archetype_total: Counter[str] = Counter()
    archetype_success: Counter[str] = Counter()
    archetype_turns: dict[str, list[int]] = defaultdict(list)

    # Per-ticker tracking
    ticker_total: Counter[str] = Counter()
    ticker_success: Counter[str] = Counter()

    # Time trends: bucket by hour
    time_buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})

    for run in runs:
        status = run.get("status", "unknown")
        best_sharpe = run.get("best_sharpe", -999.0)
        sharpe_target = run.get("sharpe_target", 1.5)
        current_turn = run.get("current_turn", 0)
        mandate = run.get("mandate_name", "")
        code_path = run.get("best_code_path", "")
        created = run.get("created_at", "")

        is_converged = status == "success" or (best_sharpe != -999.0 and best_sharpe >= sharpe_target)

        # Status counts
        if status == "success":
            success_count += 1
        elif status == "abandoned":
            abandoned_count += 1
        elif status == "max_turns_exhausted":
            exhausted_count += 1
        elif status == "running":
            running_count += 1

        # Convergence tracking
        if is_converged:
            best_turn = run.get("best_turn", current_turn)
            if best_turn > 0:
                turns_to_converge.append(best_turn)
            if code_path:
                converged_codes.add(os.path.basename(os.path.dirname(code_path)))

        # Archetype
        arch = extract_archetype(mandate)
        if arch:
            archetype_total[arch] += 1
            if is_converged:
                archetype_success[arch] += 1
                best_turn = run.get("best_turn", current_turn)
                if best_turn > 0:
                    archetype_turns[arch].append(best_turn)

        # Ticker
        tickers = extract_tickers(run)
        for t in tickers:
            ticker_total[t] += 1
            if is_converged:
                ticker_success[t] += 1

        # Time bucket (by date)
        if created:
            try:
                dt = datetime.fromisoformat(created)
                bucket = dt.strftime("%Y-%m-%d")
                time_buckets[bucket]["total"] += 1
                if is_converged:
                    time_buckets[bucket]["success"] += 1
            except (ValueError, TypeError):
                pass

    # Promotion rate
    if promoted_codes is not None:
        promoted_from_converged = len(promoted_codes & converged_codes)
        promotion_rate = (promoted_from_converged / len(converged_codes) * 100) if converged_codes else 0.0
    else:
        promoted_from_converged = 0
        promotion_rate = 0.0

    # Failure modes
    failure_modes = collect_failure_modes(runs)

    # Build archetype summary
    by_archetype: dict[str, Any] = {}
    for arch in KNOWN_ARCHETYPES:
        t = archetype_total.get(arch, 0)
        s = archetype_success.get(arch, 0)
        avg_t = (sum(archetype_turns.get(arch, [])) / len(archetype_turns[arch])) if archetype_turns.get(arch) else 0.0
        by_archetype[arch] = {
            "total": t,
            "success": s,
            "rate": (s / t * 100) if t > 0 else 0.0,
            "avg_turns": round(avg_t, 1),
        }

    # Any extra archetypes not in KNOWN list
    for arch in sorted(archetype_total):
        if arch not in KNOWN_ARCHETYPES:
            t = archetype_total[arch]
            s = archetype_success.get(arch, 0)
            avg_t = (sum(archetype_turns.get(arch, [])) / len(archetype_turns[arch])) if archetype_turns.get(arch) else 0.0
            by_archetype[arch] = {
                "total": t,
                "success": s,
                "rate": (s / t * 100) if t > 0 else 0.0,
                "avg_turns": round(avg_t, 1),
            }

    # Build ticker summary
    by_ticker: dict[str, Any] = {}
    for ticker in sorted(ticker_total):
        t = ticker_total[ticker]
        s = ticker_success.get(ticker, 0)
        by_ticker[ticker] = {
            "total": t,
            "success": s,
            "rate": (s / t * 100) if t > 0 else 0.0,
        }

    # Build time trends
    time_trends: dict[str, Any] = {}
    for bucket in sorted(time_buckets):
        b = time_buckets[bucket]
        time_trends[bucket] = {
            "total": b["total"],
            "success": b["success"],
            "rate": (b["success"] / b["total"] * 100) if b["total"] > 0 else 0.0,
        }

    avg_turns = (sum(turns_to_converge) / len(turns_to_converge)) if turns_to_converge else 0.0
    convergence_count = success_count  # primary: status == "success"

    return {
        "total_mandates": total,
        "converged": convergence_count,
        "convergence_rate": round(convergence_count / total * 100, 1) if total else 0.0,
        "avg_turns_to_converge": round(avg_turns, 1),
        "abandoned": abandoned_count,
        "max_turns_exhausted": exhausted_count,
        "success": success_count,
        "running": running_count,
        "abandonment_rate": round(abandoned_count / total * 100, 1) if total else 0.0,
        "exhaustion_rate": round(exhausted_count / total * 100, 1) if total else 0.0,
        "by_archetype": by_archetype,
        "by_ticker": by_ticker,
        "failure_modes": dict(failure_modes.most_common()),
        "promotion_rate": round(promotion_rate, 1),
        "promoted_from_converged": promoted_from_converged,
        "time_trends": time_trends,
    }


def format_report(report: dict[str, Any]) -> str:
    """Format report as a human-readable table."""
    lines: list[str] = []
    w = 80

    def hr(char="="):
        return char * w

    lines.append("CrabQuant Convergence Report")
    lines.append(hr())
    lines.append(f"Total mandates:         {report['total_mandates']}")
    lines.append(
        f"Convergence rate:       {report['convergence_rate']}% ({report['converged']}/{report['total_mandates']})"
    )
    lines.append(f"Avg turns to converge:  {report['avg_turns_to_converge']}")
    lines.append(
        f"Abandonment rate:       {report['abandonment_rate']}% ({report['abandoned']}/{report['total_mandates']})"
    )
    lines.append(
        f"Exhaustion rate:        {report['exhaustion_rate']}% ({report['max_turns_exhausted']}/{report['total_mandates']})"
    )
    lines.append(f"Promotion rate:         {report['promotion_rate']}% ({report['promoted_from_converged']} promoted from converged)")
    if report["running"]:
        lines.append(f"Still running:          {report['running']}")

    # By Archetype
    lines.append("")
    lines.append("By Archetype:")
    arch_data = report.get("by_archetype", {})
    if arch_data:
        # Column widths
        name_w = max(len(k) for k in arch_data)
        for arch, info in arch_data.items():
            t = info["total"]
            s = info["success"]
            rate = info["rate"]
            avg = info["avg_turns"]
            suffix = f" — avg {avg} turns" if s > 0 else ""
            lines.append(f"  {arch:>{name_w}}: {s:>3}/{t:<3} ({rate:>5.1f}%){suffix}")
    else:
        lines.append("  (no archetype data)")

    # By Ticker
    lines.append("")
    lines.append("By Ticker:")
    ticker_data = report.get("by_ticker", {})
    if ticker_data:
        name_w = max(len(k) for k in ticker_data)
        for ticker, info in ticker_data.items():
            t = info["total"]
            s = info["success"]
            rate = info["rate"]
            lines.append(f"  {ticker:>{name_w}}: {s:>3}/{t:<3} ({rate:>5.1f}%)")
    else:
        lines.append("  (no ticker data)")

    # Failure Modes
    lines.append("")
    lines.append("Common Failure Modes:")
    modes = report.get("failure_modes", {})
    if modes:
        for mode, count in list(modes.items())[:10]:
            lines.append(f"  {mode}: {count}")
    else:
        lines.append("  (no failures)")

    # Time Trends
    lines.append("")
    lines.append("Time Trends:")
    trends = report.get("time_trends", {})
    if trends:
        for bucket, info in trends.items():
            lines.append(f"  {bucket}: {info['success']}/{info['total']} ({info['rate']:.1f}%)")
    else:
        lines.append("  (no time data)")

    lines.append(hr())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)

    since: datetime | None = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, tzinfo=timezone.utc
        )

    runs = load_runs(runs_dir, since)
    report = compute_report(runs)

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
