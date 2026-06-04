#!/usr/bin/env python3
"""
Batch Walk-Forward Validation for Winners Without WF Data
==========================================================

Reads results/winners/winners.json, identifies entries that are missing
walk-forward validation data, and runs rolling_walk_forward() on them.

Only processes winners whose strategy name exists in the production registry
(strategies/production/registry.json) AND is importable as a module under
crabquant.strategies.<name>.

Usage:
    python scripts/batch_wf_validate_winners.py           # run all
    python scripts/batch_wf_validate_winners.py --dry-run  # show what would be done
    python scripts/batch_wf_validate_winners.py --limit 5  # only process 5 entries
"""

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

# Ensure local crabquant package is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────
WINNERS_PATH = _REPO_ROOT / "results" / "winners" / "winners.json"
REGISTRY_PATH = _REPO_ROOT / "strategies" / "production" / "registry.json"


def load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def has_wf_validation(entry: dict) -> bool:
    """Return True if entry already has walk-forward validation data."""
    val = entry.get("validation")
    if not val or not isinstance(val, dict):
        return False
    wf = val.get("walk_forward")
    if wf and isinstance(wf, dict):
        return True
    # Some entries store WF results at the validation level directly
    if val.get("walk_forward_robust") is not None:
        return True
    return False


def get_registry_strategy_names(registry: list[dict]) -> set[str]:
    """Extract unique strategy names from the production registry."""
    return {r.get("strategy_name") for r in registry if r.get("strategy_name")}


def resolve_strategy_fn(strategy_name: str, cache: dict):
    """Import strategy module and return its generate_signals function.

    Returns (fn, error_string).  Results are cached.
    """
    if strategy_name in cache:
        return cache[strategy_name]

    try:
        mod = importlib.import_module(f"crabquant.strategies.{strategy_name}")
        fn = getattr(mod, "generate_signals", None)
        if fn is None:
            cache[strategy_name] = (None, f"no generate_signals in {strategy_name}")
        else:
            cache[strategy_name] = (fn, None)
    except ModuleNotFoundError:
        cache[strategy_name] = (None, f"module crabquant.strategies.{strategy_name} not found")
    except Exception as e:
        cache[strategy_name] = (None, f"import error: {e}")

    return cache[strategy_name]


def run_rolling_wf(strategy_fn, ticker: str, params: dict) -> dict:
    """Run rolling_walk_forward with VALIDATION_CONFIG thresholds.

    Returns a dict with results or error info.
    """
    from crabquant.validation import rolling_walk_forward
    from crabquant.refinement.config import VALIDATION_CONFIG

    rcfg = {
        "train_window": VALIDATION_CONFIG.get("train_window", "18mo"),
        "test_window": VALIDATION_CONFIG.get("test_window", "6mo"),
        "step": VALIDATION_CONFIG.get("step", "6mo"),
        "min_avg_test_sharpe": VALIDATION_CONFIG.get("min_avg_test_sharpe", 0.4),
        "min_windows_passed": VALIDATION_CONFIG.get("min_windows_passed", 3),
        "min_window_test_sharpe": VALIDATION_CONFIG.get("min_window_test_sharpe", 0.1),
        "max_window_degradation": VALIDATION_CONFIG.get("max_window_degradation", 0.8),
    }
    # Allow the "rolling" sub-dict to override
    rcfg.update(VALIDATION_CONFIG.get("rolling", {}))

    start = time.time()
    try:
        rwf = rolling_walk_forward(
            strategy_fn,
            ticker,
            params,
            train_window=rcfg["train_window"],
            test_window=rcfg["test_window"],
            step=rcfg["step"],
            min_avg_test_sharpe=rcfg["min_avg_test_sharpe"],
            min_windows_passed=rcfg["min_windows_passed"],
            min_window_test_sharpe=rcfg["min_window_test_sharpe"],
            max_window_degradation=rcfg["max_window_degradation"],
        )
        elapsed = time.time() - start
        return {
            "success": True,
            "robust": rwf.robust,
            "avg_test_sharpe": rwf.avg_test_sharpe,
            "min_test_sharpe": rwf.min_test_sharpe,
            "avg_degradation": rwf.avg_degradation,
            "num_windows": rwf.num_windows,
            "windows_passed": rwf.windows_passed,
            "notes": rwf.notes,
            "window_results": rwf.window_results,
            "elapsed": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed": round(elapsed, 1),
        }


def build_validation_dict(rwf_result: dict) -> dict:
    """Convert rolling_walk_forward result into the validation dict format
    used in winners.json (matching the format from promotion.py).
    """
    if not rwf_result["success"]:
        return {
            "passed": False,
            "walk_forward_robust": False,
            "walk_forward": None,
            "error": rwf_result.get("error"),
        }

    return {
        "passed": rwf_result["robust"],
        "walk_forward_robust": rwf_result["robust"],
        "walk_forward": {
            "avg_test_sharpe": rwf_result["avg_test_sharpe"],
            "min_test_sharpe": rwf_result["min_test_sharpe"],
            "avg_degradation": rwf_result["avg_degradation"],
            "num_windows": rwf_result["num_windows"],
            "windows_passed": rwf_result["windows_passed"],
            "robust": rwf_result["robust"],
            "notes": rwf_result["notes"],
            "window_results": rwf_result["window_results"],
        },
        "cross_ticker_robust": None,
        "cross_ticker": None,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch walk-forward validation for winners without WF data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be validated without actually running",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process N entries (0 = all)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  CrabQuant — Batch WF Validation for Winners Without WF Data")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────────
    if not WINNERS_PATH.exists():
        print(f"\nERROR: winners.json not found at {WINNERS_PATH}")
        return 1

    winners = load_json(WINNERS_PATH)
    print(f"\nLoaded {len(winners)} winners from {WINNERS_PATH.relative_to(_REPO_ROOT)}")

    registry = load_json(REGISTRY_PATH) if REGISTRY_PATH.exists() else []
    registry_names = get_registry_strategy_names(registry)
    print(f"Loaded {len(registry_names)} unique strategy names from registry")

    # ── Identify entries without WF data ───────────────────────────────────
    missing_wf = [w for w in winners if not has_wf_validation(w)]
    print(f"\nEntries missing WF validation: {len(missing_wf)} / {len(winners)}")

    if not missing_wf:
        print("\nAll winners already have WF validation data. Nothing to do.")
        return 0

    # ── Categorise by strategy ─────────────────────────────────────────────
    unique_strats = sorted(set(w["strategy"] for w in missing_wf))
    print(f"\nUnique strategies in missing-WF entries: {len(unique_strats)}")
    for s in unique_strats:
        count = sum(1 for w in missing_wf if w["strategy"] == s)
        in_reg = "in registry" if s in registry_names else "NOT in registry"
        print(f"  {s}: {count} entries ({in_reg})")

    # ── Filter to resolvable strategies ────────────────────────────────────
    # Primary: strategy in registry.  Secondary: strategy module importable
    # even if not in registry (e.g., rsi_crossover, informed_simple_adaptive).
    candidates = []
    skipped_no_registry = []
    module_preload: dict = {}

    for w in missing_wf:
        sname = w["strategy"]
        if sname in registry_names:
            candidates.append(w)
        else:
            # Try importing as fallback — some strategies exist as modules
            # but were never registered (e.g., core library strategies).
            fn, err = resolve_strategy_fn(sname, module_preload)
            if fn is not None:
                candidates.append(w)
            else:
                skipped_no_registry.append(w)

    print(f"\nCandidates (registry or importable): {len(candidates)}")
    print(f"Skipped (not in registry & not importable): {len(skipped_no_registry)}")

    if not candidates:
        print("\nNo candidates to validate. All missing-WF entries use non-registry strategies.")
        return 0

    # Apply limit
    if args.limit > 0:
        candidates = candidates[: args.limit]
        print(f"\n--limit {args.limit} applied: processing {len(candidates)} entries")

    # ── Dry-run mode ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n" + "-" * 72)
        print("  DRY RUN — would validate the following entries:")
        print("-" * 72)
        for i, w in enumerate(candidates, 1):
            print(
                f"  [{i:3d}] {w['strategy']:40s} | {w['ticker']:5s} | "
                f"sharpe={w.get('sharpe', 0):.3f}"
            )
        print(f"\n  Total: {len(candidates)} entries would be validated")
        print("  Use without --dry-run to actually run validation")
        return 0

    # ── Import strategy modules (with cache) ───────────────────────────────
    module_cache: dict = {}
    importable = 0
    for w in candidates:
        fn, err = resolve_strategy_fn(w["strategy"], module_cache)
        if fn is not None:
            importable += 1

    print(f"\nImportable strategies: {importable} / {len(candidates)}")

    # ── Run validation ─────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  Running Rolling Walk-Forward Validation")
    print("-" * 72)

    results_summary = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
    winners_by_key = {w.get("key", ""): i for i, w in enumerate(winners)}
    updated_count = 0

    for i, w in enumerate(candidates, 1):
        strategy = w["strategy"]
        ticker = w["ticker"]
        params = w.get("params", {})
        sharpe = w.get("sharpe", 0)
        label = f"{strategy}|{ticker}"

        print(
            f"  [{i:3d}/{len(candidates)}] {label:50s} | sharpe={sharpe:.3f} ... ",
            end="",
            flush=True,
        )

        fn, err = resolve_strategy_fn(strategy, module_cache)
        if fn is None:
            print(f"SKIP ({err})")
            results_summary["skip"] += 1
            continue

        rwf_result = run_rolling_wf(fn, ticker, params)

        if not rwf_result["success"]:
            print(f"ERROR ({rwf_result.get('error', 'unknown')[:60]})")
            results_summary["error"] += 1
            # Still store the error in validation
            val_dict = build_validation_dict(rwf_result)
        else:
            status = "PASS" if rwf_result["robust"] else "FAIL"
            print(
                f"{status} | avg_sharpe={rwf_result['avg_test_sharpe']:.3f} | "
                f"windows={rwf_result['windows_passed']}/{rwf_result['num_windows']} | "
                f"degrad={rwf_result['avg_degradation']:.3f} | "
                f"{rwf_result['elapsed']:.1f}s"
            )
            results_summary["pass" if rwf_result["robust"] else "fail"] += 1
            val_dict = build_validation_dict(rwf_result)

        # Update the winner entry
        key = w.get("key", "")
        idx = winners_by_key.get(key)
        if idx is not None:
            winners[idx]["validation"] = val_dict
            updated_count += 1

    # ── Save updated winners.json ──────────────────────────────────────────
    if updated_count > 0:
        save_json(WINNERS_PATH, winners)
        print(f"\n✅ Updated {updated_count} entries in {WINNERS_PATH.relative_to(_REPO_ROOT)}")
    else:
        print("\n⚠ No entries were updated")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"\n  Total winners:               {len(winners)}")
    print(f"  Previously missing WF:       {len(missing_wf)}")
    print(f"  Candidates (in registry):    {len(candidates)}")
    print(f"  Skipped (no registry match): {len(skipped_no_registry)}")
    print(f"\n  Validation results:")
    print(f"    PASS (robust):  {results_summary['pass']}")
    print(f"    FAIL:           {results_summary['fail']}")
    print(f"    ERROR:          {results_summary['error']}")
    print(f"    SKIP (no code): {results_summary['skip']}")
    print(f"    Updated in JSON: {updated_count}")

    still_missing = sum(1 for w in winners if not has_wf_validation(w))
    print(f"\n  Remaining without WF: {still_missing}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
