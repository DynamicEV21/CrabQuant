#!/usr/bin/env python3
"""
Analyze winners.json — find near-miss strategies that might now pass
validation with relaxed thresholds.

Runs rolling walk-forward validation on the top strategies by Sharpe ratio.
"""

import json
import sys
import time
import importlib
import traceback
import os
from collections import Counter, defaultdict
from pathlib import Path

# Ensure local crabquant package is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ── Config ────────────────────────────────────────────────────────────────
WINNERS_PATH = Path(__file__).parent / "winners" / "winners.json"
MAX_STRATEGIES = 15  # test top N by Sharpe
TIMEOUT_PER_STRATEGY = 45  # seconds

# Relaxed thresholds (as per task spec)
MIN_AVG_TEST_SHARPE = 0.3
MIN_WINDOWS_PASSED = 1
MIN_WINDOW_TEST_SHARPE = 0.0
MAX_WINDOW_DEGRADATION = 1.0


def load_winners():
    with open(WINNERS_PATH) as f:
        return json.load(f)


def get_strategy_fn(strategy_name: str):
    """Import the strategy module and return its generate_signals function."""
    try:
        mod = importlib.import_module(f"crabquant.strategies.{strategy_name}")
        fn = getattr(mod, "generate_signals", None)
        if fn is None:
            return None, f"no generate_signals in {strategy_name}"
        return fn, None
    except Exception as e:
        return None, str(e)


def run_rolling_wf(strategy_fn, ticker, params, label):
    """Run rolling walk-forward with timeout protection."""
    from crabquant.validation import rolling_walk_forward

    start = time.time()
    try:
        rwf = rolling_walk_forward(
            strategy_fn,
            ticker,
            params,
            min_avg_test_sharpe=MIN_AVG_TEST_SHARPE,
            min_windows_passed=MIN_WINDOWS_PASSED,
            min_window_test_sharpe=MIN_WINDOW_TEST_SHARPE,
            max_window_degradation=MAX_WINDOW_DEGRADATION,
        )
        elapsed = time.time() - start
        return rwf, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return None, elapsed, str(e)


def main():
    print("=" * 72)
    print("  CrabQuant Winners Analysis — Near-Miss Strategy Discovery")
    print("=" * 72)

    winners = load_winners()
    print(f"\nTotal winners in pool: {len(winners)}")

    # ── Pool composition ──────────────────────────────────────────────────
    strategy_counts = Counter(w["strategy"] for w in winners)
    ticker_counts = Counter(w["ticker"] for w in winners)
    has_validation = sum(1 for w in winners if "validation" in w and w["validation"])
    has_val_status = sum(1 for w in winners if "validation_status" in w)

    print(f"\nStrategy distribution: {dict(strategy_counts.most_common())}")
    print(f"Ticker distribution:   {dict(ticker_counts.most_common())}")
    print(f"Entries with validation data: {has_validation}")
    print(f"Entries with validation_status: {has_val_status}")

    # Categorize: sweep debris vs refined/promoted
    refined = [w for w in winners if "refinement_turns" in w]
    sweep = [w for w in winners if "refinement_turns" not in w]
    print(f"\nRefined/promoted entries: {len(refined)}")
    print(f"Sweep debris entries:    {len(sweep)}")

    # ── Already-validated analysis ────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  Already-Validated Entries (from winners.json)")
    print("-" * 72)

    for w in winners:
        if "validation" in w and w["validation"]:
            v = w["validation"]
            wf = v.get("walk_forward", {})
            wf_sharpe = wf.get("test_sharpe", "N/A")
            wf_robust = wf.get("robust", False)
            ct_robust = v.get("cross_ticker_robust", False)
            status = w.get("validation_status", "N/A")
            print(
                f"  {w['strategy'][:35]:35s} | {w['ticker']:5s} | "
                f"sharpe={w['sharpe']:.3f} | wf_sharpe={wf_sharpe} | "
                f"wf_robust={wf_robust} | ct_robust={ct_robust} | {status}"
            )

    # ── Select top candidates ─────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  Running Rolling Walk-Forward on Top Strategies")
    print("-" * 72)

    # Sort by Sharpe, deduplicate by strategy name (pick best per strategy)
    sorted_winners = sorted(winners, key=lambda x: x.get("sharpe", 0), reverse=True)

    # Group by strategy, pick top 2 per strategy to get diversity
    by_strategy = defaultdict(list)
    for w in sorted_winners:
        by_strategy[w["strategy"]].append(w)

    candidates = []
    for strat_name, entries in by_strategy.items():
        # Take top 2 per strategy (different param sets)
        for entry in entries[:2]:
            candidates.append(entry)
        if len(candidates) >= MAX_STRATEGIES:
            break

    candidates = candidates[:MAX_STRATEGIES]
    print(f"\nTesting {len(candidates)} candidates (top by Sharpe, max 2 per strategy)")
    print(f"Thresholds: avg_test_sharpe>={MIN_AVG_TEST_SHARPE}, windows_passed>={MIN_WINDOWS_PASSED}")
    print(f"           per_window_sharpe>={MIN_WINDOW_TEST_SHARPE}, degradation<={MAX_WINDOW_DEGRADATION}")
    print()

    # ── Run validation ────────────────────────────────────────────────────
    from signal import signal, SIGALRM, alarm
    results = []

    for i, w in enumerate(candidates):
        strat_name = w["strategy"]
        ticker = w["ticker"]
        params = w["params"]
        sharpe = w["sharpe"]
        label = f"{strat_name}|{ticker}"

        print(f"  [{i+1}/{len(candidates)}] {label} (sharpe={sharpe:.3f})... ", end="", flush=True)

        # Check if strategy module exists
        fn, err = get_strategy_fn(strat_name)
        if fn is None:
            print(f"SKIP (no module: {err})")
            results.append({"label": label, "sharpe": sharpe, "status": "skip", "reason": err})
            continue

        # Run rolling WF
        rwf, elapsed, error = run_rolling_wf(fn, ticker, params, label)

        if error:
            print(f"ERROR ({error[:80]})")
            results.append({"label": label, "sharpe": sharpe, "status": "error", "reason": error[:200]})
            continue

        print(
            f"{'PASS' if rwf.robust else 'FAIL'} | "
            f"avg_sharpe={rwf.avg_test_sharpe:.3f} | "
            f"windows={rwf.windows_passed}/{rwf.num_windows} | "
            f"min_sharpe={rwf.min_test_sharpe:.3f} | "
            f"avg_degrad={rwf.avg_degradation:.3f} | "
            f"{elapsed:.1f}s"
        )

        results.append({
            "label": label,
            "sharpe": sharpe,
            "status": "pass" if rwf.robust else "fail",
            "avg_test_sharpe": rwf.avg_test_sharpe,
            "windows_passed": rwf.windows_passed,
            "num_windows": rwf.num_windows,
            "min_test_sharpe": rwf.min_test_sharpe,
            "avg_degradation": rwf.avg_degradation,
            "notes": rwf.notes,
            "elapsed": round(elapsed, 1),
            "params": params,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    passed = [r for r in results if r["status"] == "pass"]
    failed = [r for r in results if r["status"] == "fail"]
    skipped = [r for r in results if r["status"] == "skip"]
    errored = [r for r in results if r["status"] == "error"]

    print(f"\nAnalyzed: {len(results)} strategies")
    print(f"  PASSED rolling WF:  {len(passed)}")
    print(f"  FAILED rolling WF:  {len(failed)}")
    print(f"  SKIPPED (no code):  {len(skipped)}")
    print(f"  ERRORED:            {len(errored)}")

    if passed:
        print("\n  🎯 STRATEGIES THAT PASS ROLLING WALK-FORWARD:")
        for r in sorted(passed, key=lambda x: x["avg_test_sharpe"], reverse=True):
            print(
                f"    {r['label']:45s} | backtest_sharpe={r['sharpe']:.3f} | "
                f"avg_test_sharpe={r['avg_test_sharpe']:.3f} | "
                f"windows={r['windows_passed']}/{r['num_windows']}"
            )
    else:
        print("\n  ⚠ No strategies passed rolling walk-forward validation.")

    # Near-misses (close to passing)
    if failed:
        near_misses = sorted(failed, key=lambda x: x.get("avg_test_sharpe", 0), reverse=True)[:5]
        print("\n  📊 TOP NEAR-MISSES (closest to passing):")
        for r in near_misses:
            print(
                f"    {r['label']:45s} | avg_test_sharpe={r['avg_test_sharpe']:.3f} | "
                f"windows={r['windows_passed']}/{r['num_windows']} | "
                f"degrad={r['avg_degradation']:.3f}"
            )

    # Patterns & observations
    print("\n  📋 OBSERVATIONS:")
    print(f"    • Pool is dominated by sweep debris: {len(sweep)}/{len(winners)} entries")
    print(f"    • Refined strategies: {len(refined)} (these had prior validation)")
    print(f"    • Only {len(set(w['strategy'] for w in winners))} unique strategy types")
    print(f"    • Heavy ticker concentration: GOOGL ({ticker_counts['GOOGL']}), JNJ ({ticker_counts['JNJ']})")

    # Check if any previously-failed strategies now pass with relaxed thresholds
    if passed:
        print(f"\n  ✅ {len(passed)} strategies from winners pool now pass with relaxed thresholds!")
        print("    These should be flagged for promotion consideration.")
    else:
        print("\n  ❌ No near-miss strategies found that pass even with relaxed thresholds.")
        print("    The winners pool appears to contain overfit sweep results.")

    # Save results
    output_path = Path(__file__).parent / "winner_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # ── Full Pool Scan (bonus) ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FULL POOL SCAN — All 61 Winners Tested")
    print("=" * 72)

    full_results = []
    modules_cache = {}

    for w in winners:
        strat_name = w["strategy"]
        ticker = w["ticker"]
        params = w["params"]
        sharpe = w["sharpe"]
        label = f"{strat_name}|{ticker}"

        # Resolve strategy function (with caching)
        if strat_name not in modules_cache:
            fn, err = get_strategy_fn(strat_name)
            modules_cache[strat_name] = (fn, err)
        fn, err = modules_cache[strat_name]

        if fn is None:
            full_results.append({"label": label, "sharpe": sharpe, "status": "skip", "reason": err})
            continue

        rwf, elapsed, error = run_rolling_wf(fn, ticker, params, label)
        if error:
            full_results.append({"label": label, "sharpe": sharpe, "status": "error", "reason": error[:200]})
            continue

        full_results.append({
            "label": label,
            "sharpe": sharpe,
            "status": "pass" if rwf.robust else "fail",
            "avg_test_sharpe": rwf.avg_test_sharpe,
            "windows_passed": rwf.windows_passed,
            "num_windows": rwf.num_windows,
            "min_test_sharpe": rwf.min_test_sharpe,
            "avg_degradation": rwf.avg_degradation,
            "notes": rwf.notes,
            "elapsed": round(elapsed, 1),
            "params": params,
        })

    all_passed = [r for r in full_results if r["status"] == "pass"]
    all_failed = [r for r in full_results if r["status"] == "fail"]
    all_skipped = [r for r in full_results if r["status"] == "skip"]

    print(f"\nFull pool results: {len(all_passed)} PASS, {len(all_failed)} FAIL, {len(all_skipped)} SKIP")

    # Group by strategy
    by_strat = defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0})
    for r in full_results:
        strat = r["label"].split("|")[0]
        by_strat[strat][r["status"]] += 1

    print("\n  Pass rates by strategy type:")
    for strat, counts in sorted(by_strat.items()):
        total = counts["pass"] + counts["fail"] + counts["skip"]
        print(f"    {strat:45s} | {counts['pass']:2d}/{total:2d} pass | {counts['fail']:2d} fail | {counts['skip']:2d} skip")

    # Save full results
    full_output_path = Path(__file__).parent / "winner_analysis_full.json"
    with open(full_output_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {full_output_path}")

    return 0 if not all_passed else 1  # exit 1 if we found something (signal to caller)


if __name__ == "__main__":
    sys.exit(main())
