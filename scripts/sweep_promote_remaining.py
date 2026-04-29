#!/usr/bin/env python3
"""
Sweep Promote Remaining — Cycle 13

Takes the 113 validated strategy-ticker pairs from
results/sweep_results_cycle13.json and promotes any that are NOT already
in results/winners/winners.json with validation_status=promoted.

For each newly promoted entry:
  1. Appends a winner entry to winners.json (matching batch_cycle13 format)
  2. Copies the base strategy .py file to a ticker-specific filename
     (e.g., crabquant/strategies/adx_pullback.py → adx_pullback_spy.py)
     if the ticker-specific file doesn't already exist

Usage:
    python scripts/sweep_promote_remaining.py              # Dry-run (no writes)
    python scripts/sweep_promote_remaining.py --execute     # Actually promote

Designed to be run from the project root (or any cwd that has results/ and
crabquant/ as siblings).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_FILE = PROJECT_ROOT / "results" / "sweep_results_cycle13.json"
WINNERS_FILE = PROJECT_ROOT / "results" / "winners" / "winners.json"
STRATEGIES_DIR = PROJECT_ROOT / "crabquant" / "strategies"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list | dict:
    """Load JSON file, returning empty list on missing / parse error."""
    if not path.exists():
        print(f"  [WARN] File not found: {path}")
        return []
    try:
        text = path.read_text()
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parse error in {path}: {e}")
        return []


def save_json(path: Path, data) -> None:
    """Write JSON with 2-space indent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_winner_entry(sweep_entry: dict) -> dict:
    """Convert a sweep_results 'passed' entry into a winners.json entry.

    Matches the format used by the existing batch_cycle13 promotions:
    - strategy = registry_name (e.g. "adx_pullback_spy")
    - sharpe/return/max_drawdown/trades = 0 (not available from sweep)
    - validation object with sweep-derived metrics
    - refinement_run = "batch_cycle13", refinement_turns = 0
    """
    return {
        "strategy": sweep_entry["registry_name"],
        "ticker": sweep_entry["ticker"],
        "sharpe": round(sweep_entry["wf_avg_test_sharpe"], 3),
        "return": 0,
        "max_drawdown": 0,
        "trades": 0,
        "params": sweep_entry["params"],
        "refinement_run": "batch_cycle13",
        "refinement_turns": 0,
        "validation": {
            "passed": True,
            "walk_forward_robust": True,
            "cross_ticker_robust": True,
            "validation_method": "rolling",
            "sweep_wf_avg_test_sharpe": round(sweep_entry["wf_avg_test_sharpe"], 3),
            "sweep_wf_windows": sweep_entry["wf_windows"],
            "sweep_ct_avg_sharpe": round(sweep_entry["ct_avg_sharpe"], 3),
            "sweep_ct_profitable": sweep_entry.get("ct_profitable", ""),
        },
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": "promoted",
        "regime_tags": None,
    }


def copy_strategy_file(sweep_entry: dict, dry_run: bool = False) -> bool:
    """Copy base strategy .py to ticker-specific filename if missing.

    The promotion pipeline creates ticker-specific copies (e.g.,
    adx_pullback.py → adx_pullback_spy.py) so each promoted strategy-ticker
    pair has its own importable module.  The content is identical — the
    ticker-specific params are stored in winners.json, not in the .py file.

    Returns True if the file was created (or would be in dry-run).
    """
    base_path = Path(sweep_entry["strategy_code_path"])
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path

    ticker_name = f"{sweep_entry['registry_name']}.py"
    target_path = STRATEGIES_DIR / ticker_name

    if target_path.exists():
        return False

    if not base_path.exists():
        print(f"    [ERROR] Base strategy not found: {base_path}")
        return False

    if dry_run:
        print(f"    [WOULD COPY] {base_path.name} → {ticker_name}")
        return True

    shutil.copy2(base_path, target_path)
    print(f"    [COPIED] {base_path.name} → {ticker_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Promote remaining validated sweep entries to winners.json"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually write changes (default is dry-run)",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    mode_label = "DRY-RUN" if dry_run else "EXECUTE"
    print(f"=== Sweep Promote Remaining — {mode_label} ===\n")

    # ------------------------------------------------------------------
    # 1. Load sweep results
    # ------------------------------------------------------------------
    sweep_data = load_json(SWEEP_FILE)
    if not isinstance(sweep_data, dict) or "passed" not in sweep_data:
        print("[FATAL] Cannot load sweep_results_cycle13.json or missing 'passed' key")
        sys.exit(1)

    passed_entries = sweep_data["passed"]
    print(f"Sweep validated pairs: {len(passed_entries)}")

    # ------------------------------------------------------------------
    # 2. Load existing winners
    # ------------------------------------------------------------------
    winners = load_json(WINNERS_FILE)
    if not isinstance(winners, list):
        winners = []

    # Build set of already-promoted strategy names (keyed by strategy field)
    promoted_names: set[str] = set()
    for w in winners:
        if w.get("validation_status") == "promoted":
            promoted_names.add(w["strategy"])

    print(f"Existing winners: {len(winners)}")
    print(f"Already promoted: {len(promoted_names)}\n")

    # ------------------------------------------------------------------
    # 3. Identify missing entries (sweep pairs not yet promoted)
    # ------------------------------------------------------------------
    to_promote = []
    for entry in passed_entries:
        name = entry["registry_name"]
        if name not in promoted_names:
            to_promote.append(entry)

    if not to_promote:
        print("✓ All sweep entries are already promoted — nothing to do.")
        return

    print(f"Entries to promote: {len(to_promote)}\n")

    # Group by base strategy for summary
    from collections import Counter
    base_counts = Counter(e["strategy"] for e in to_promote)
    print("By base strategy:")
    for base, count in base_counts.most_common():
        print(f"  {base}: {count} ticker pairs")
    print()

    # ------------------------------------------------------------------
    # 4. Promote each missing entry
    # ------------------------------------------------------------------
    winners_added = 0
    files_created = 0
    files_skipped = 0
    errors = 0

    for entry in to_promote:
        name = entry["registry_name"]
        ticker = entry["ticker"]
        base_strategy = entry["strategy"]
        print(f"  [{name}] ticker={ticker}  wf_sharpe={entry['wf_avg_test_sharpe']:.3f}  "
              f"ct_sharpe={entry['ct_avg_sharpe']:.3f}  windows={entry['wf_windows']}")

        # Build winner entry
        winner_entry = build_winner_entry(entry)

        if dry_run:
            print(f"    [WOULD ADD] to winners.json")
            winners_added += 1
        else:
            winners.append(winner_entry)
            winners_added += 1

        # Copy/create strategy .py file
        created = copy_strategy_file(entry, dry_run=dry_run)
        if created:
            files_created += 1
        else:
            files_skipped += 1

    # ------------------------------------------------------------------
    # 5. Save winners.json (only in execute mode)
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Winners entries to add:  {winners_added}")
    print(f"  Strategy files created:  {files_created}")
    print(f"  Strategy files skipped:  {files_skipped} (already exist)")
    print(f"  Errors:                 {errors}")

    if dry_run:
        print(f"\n  Mode: DRY-RUN — no files were modified.")
        print(f"  Re-run with --execute to apply changes.")
    else:
        save_json(WINNERS_FILE, winners)
        print(f"\n  Mode: EXECUTE — wrote {len(winners)} entries to {WINNERS_FILE}")
        print(f"  Strategy files in {STRATEGIES_DIR}/")

    # ------------------------------------------------------------------
    # 6. Optionally: report on near_miss entries
    # ------------------------------------------------------------------
    near_miss = sweep_data.get("near_miss", [])
    if near_miss:
        print(f"\n  Note: {len(near_miss)} near_miss entries in sweep (not promoted)")
        print(f"  These can be reviewed with --near-miss flag (future enhancement)")


if __name__ == "__main__":
    main()
